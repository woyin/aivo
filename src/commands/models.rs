//! ModelsCommand handler for listing available models from the active provider.
//! Calls provider-specific model listing endpoints (OpenAI, Gemini, Cloudflare).
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::time::{Duration, Instant};

use crate::commands::normalize_base_url;
use crate::errors::ExitCode;
use crate::services::ai_launcher::AIToolType;
use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INTEGRATION_ID, CopilotTokenManager,
};
use crate::services::http_debug::LoggedSend;
use crate::services::http_utils;
use crate::services::models_cache::{ModelMetadata, ModelsCache, full_catalog_key};
use crate::services::provider_profile::{
    ModelListingStrategy, cloudflare_ai_base, is_aivo_starter_base, provider_profile_for_key,
};
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

pub struct ModelsCommand {
    session_store: SessionStore,
    cache: ModelsCache,
}

/// Rich model information for display. Fields are populated from API metadata
/// when the provider returns them.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ModelInfo {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Raw context window; not exposed via `--json`.
    #[serde(skip)]
    pub context_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output: Option<String>,
    /// Raw max-output tokens; not exposed via `--json`.
    #[serde(skip)]
    pub max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiplier: Option<f64>,
    /// Marked deprecated by models.dev; display-only, never cached.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub deprecated: bool,
    /// Reasoning-effort levels advertised by `/v1/models`; drives `/effort` for
    /// catalog-only models (e.g. `aivo/starter`) the snapshot lacks.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub reasoning_efforts: Vec<String>,
}

impl ModelInfo {
    fn id_only(id: String) -> Self {
        Self {
            id,
            context: None,
            context_tokens: None,
            max_output: None,
            max_output_tokens: None,
            input_price: None,
            output_price: None,
            multiplier: None,
            deprecated: false,
            reasoning_efforts: Vec::new(),
        }
    }
}

#[derive(Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModel>,
}

/// Loosely-parsed model entry. Only `id` is required; everything else is
/// extracted by searching field names for known patterns so we adapt to
/// any provider's response shape (OpenRouter, Vercel, OpenAI, etc.).
#[derive(Deserialize)]
struct OpenAIModel {
    id: String,
    #[serde(flatten)]
    extra: serde_json::Map<String, Value>,
}

impl OpenAIModel {
    fn into_model_info(self) -> ModelInfo {
        let context_tokens = self.find_context();
        let context = context_tokens.map(format_token_count);
        let max_output_tokens = self.find_max_output();
        let max_output = max_output_tokens.map(format_token_count);
        let (input_price, output_price) = self.find_pricing();
        let multiplier = self.find_multiplier();
        let reasoning_efforts = self.find_reasoning_efforts();
        ModelInfo {
            id: self.id,
            context,
            context_tokens,
            max_output,
            max_output_tokens,
            input_price,
            output_price,
            multiplier,
            deprecated: false,
            reasoning_efforts,
        }
    }

    /// Levels from a `reasoning_efforts` string array (exact key), lowercased;
    /// empty when absent or not a string array.
    fn find_reasoning_efforts(&self) -> Vec<String> {
        self.extra
            .get("reasoning_efforts")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Search for a Copilot-style billing multiplier (`billing.multiplier`).
    fn find_multiplier(&self) -> Option<f64> {
        let billing = self.extra.get("billing")?.as_object()?;
        billing.get("multiplier").and_then(|v| v.as_f64())
    }

    /// Search for a context-window field: context_length, context_window,
    /// context_size, max_context_length, max_input_tokens, etc.
    fn find_context(&self) -> Option<u64> {
        for (key, val) in &self.extra {
            let k = key.to_ascii_lowercase();
            if k.contains("context")
                && !k.contains("price")
                && !k.contains("cost")
                && let Some(n) = http_utils::parse_token_u64(val)
            {
                return Some(n);
            }
        }
        // Fallback: direct lookup
        self.extra
            .get("max_input_tokens")
            .and_then(http_utils::parse_token_u64)
    }

    /// Search for a max-output field: max_tokens, max_output_tokens,
    /// max_completion_tokens (top-level or nested in sub-objects).
    fn find_max_output(&self) -> Option<u64> {
        // Top-level fields
        for (key, val) in &self.extra {
            let k = key.to_ascii_lowercase();
            if (k == "max_tokens"
                || k == "max_output_tokens"
                || k == "max_completion_tokens"
                || k == "max_output")
                && !k.contains("price")
                && let Some(n) = http_utils::parse_token_u64(val)
            {
                return Some(n);
            }
        }
        // Nested in sub-objects (e.g. top_provider.max_completion_tokens)
        for (_key, val) in &self.extra {
            if let Some(obj) = val.as_object() {
                for (k2, v2) in obj {
                    let k = k2.to_ascii_lowercase();
                    if k.contains("max")
                        && (k.contains("output") || k.contains("completion") || k.contains("token"))
                        && let Some(n) = http_utils::parse_token_u64(v2)
                    {
                        return Some(n);
                    }
                }
            }
        }
        None
    }

    /// Search for pricing info in a nested "pricing"/"price" object,
    /// looking for input/prompt and output/completion fields.
    fn find_pricing(&self) -> (Option<String>, Option<String>) {
        for (key, val) in &self.extra {
            let k = key.to_ascii_lowercase();
            if (k == "pricing" || k == "price" || k == "prices")
                && let Some(obj) = val.as_object()
            {
                let input =
                    find_price_value(obj, &["input", "prompt", "input_cost", "input_price"]);
                let output = find_price_value(
                    obj,
                    &["output", "completion", "output_cost", "output_price"],
                );
                return (
                    input.and_then(|s| format_price_per_million(&s)),
                    output.and_then(|s| format_price_per_million(&s)),
                );
            }
        }
        (None, None)
    }
}

/// Search a pricing object for a field matching one of the candidate names.
fn find_price_value(obj: &serde_json::Map<String, Value>, candidates: &[&str]) -> Option<String> {
    for candidate in candidates {
        if let Some(val) = obj.get(*candidate) {
            if let Some(s) = val.as_str() {
                return Some(s.to_string());
            }
            if let Some(n) = val.as_f64() {
                return Some(n.to_string());
            }
        }
    }
    None
}

#[derive(Deserialize)]
struct GeminiModelsResponse {
    models: Vec<GeminiModel>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiModel {
    name: String,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
    #[serde(default)]
    input_token_limit: Option<u64>,
    #[serde(default)]
    output_token_limit: Option<u64>,
}

#[derive(Deserialize)]
struct CloudflareModelsResponse {
    #[serde(default)]
    result: Vec<CloudflareModel>,
    result_info: Option<CloudflareResultInfo>,
}

#[derive(Deserialize)]
struct CloudflareModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct CloudflareResultInfo {
    total_pages: Option<u32>,
}

impl ModelsCommand {
    pub fn new(session_store: SessionStore, cache: ModelsCache) -> Self {
        Self {
            session_store,
            cache,
        }
    }

    pub async fn execute(
        &self,
        key_override: Option<ApiKey>,
        refresh: bool,
        search: Option<String>,
        json: bool,
    ) -> ExitCode {
        match self
            .execute_internal(key_override, refresh, search, json)
            .await
        {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{}", style::yellow(e.to_string()));
                crate::errors::exit_code_for_error(&e)
            }
        }
    }

    async fn execute_internal(
        &self,
        key_override: Option<ApiKey>,
        refresh: bool,
        search: Option<String>,
        json: bool,
    ) -> Result<ExitCode> {
        let mut key = match key_override {
            Some(k) => k,
            None => match self.session_store.get_active_key_info().await? {
                Some(k) => k,
                None => {
                    eprintln!(
                        "{} No API key configured. Run 'aivo keys add' first.",
                        style::red("Error:")
                    );
                    return Ok(ExitCode::AuthError);
                }
            },
        };

        if key.is_any_oauth() {
            eprintln!(
                "{} Key '{}' is an OAuth credential — it doesn't have a model listing API.",
                style::red("Error:"),
                key.display_name()
            );
            eprintln!(
                "  {} Use `{}` to launch the tool directly, or switch to a regular API key with `aivo use`.",
                style::dim("hint:"),
                key.oauth_tool_hint()
            );
            return Ok(ExitCode::UserError);
        }

        if key.is_cursor_acp() {
            SessionStore::decrypt_key_secret(&mut key)?;
        }

        let is_ollama = crate::services::provider_profile::is_ollama_base(&key.base_url);
        let all_cache_key = full_catalog_cache_key_for_key(&key);

        // `aivo models` shows the provider's full catalog, including image,
        // audio, and embedding models. Chat pickers filter/annotate at their
        // own call sites. Serve from cache whenever the entry is fresh — the
        // TTL handles staleness, and id-only providers (minimax, cloudflare)
        // round-trip as id-only rows just like a fresh fetch would.
        let cached_entry = if refresh || is_ollama {
            None
        } else {
            self.cache
                .get_with_metadata(&all_cache_key)
                .await
                .filter(|(ids, _)| !cursor_cache_looks_corrupt(&key, ids))
        };

        let mut models = if let Some((ids, meta)) = cached_entry {
            models_from_cache(ids, meta)
        } else {
            SessionStore::decrypt_key_secret(&mut key)?;
            let client = http_utils::router_http_client();
            let started_at = Instant::now();
            let (spinning, spinner_handle) = style::start_spinner(Some(" Fetching models..."));
            let result = fetch_models_detailed_filtered(&client, &key, false).await;
            let min_visible = Duration::from_millis(350);
            if let Some(remaining) = min_visible.checked_sub(started_at.elapsed()) {
                tokio::time::sleep(remaining).await;
            }
            style::stop_spinner(&spinning);
            let _ = spinner_handle.await;
            let fresh = result?;

            // Cache the full list (including image/audio/embed) under the `#all`
            // namespace so `aivo image` and future broad pickers can share it.
            // Persist every column the table prints so the next call can be
            // satisfied without a network roundtrip.
            if !is_ollama {
                let ids: Vec<String> = fresh.iter().map(|m| m.id.clone()).collect();
                let metadata = build_metadata_map(&fresh);
                self.cache
                    .set_with_metadata(&all_cache_key, ids, metadata)
                    .await;
            }
            fresh
        };

        // Fill missing limit columns from the embedded models.dev snapshot so
        // id-only providers (starter, minimax, cloudflare) still show
        // context/output. Display-only: the cache write above stores what the
        // provider actually returned.
        for model in &mut models {
            enrich_from_snapshot(model);
        }

        let is_starter = is_aivo_starter_base(&key.base_url);
        // First-party key shows the account plan label, not the raw `aivo-starter` sentinel.
        let starter_account = is_starter
            .then(crate::services::account_store::load)
            .flatten();
        models.sort_by(|a, b| {
            if is_starter {
                let a_s = a.id.ends_with("/starter");
                let b_s = b.id.ends_with("/starter");
                if a_s != b_s {
                    return if a_s {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    };
                }
            }
            a.id.cmp(&b.id)
        });

        let searching = if let Some(ref query) = search {
            let q = query.trim().to_lowercase();
            if q.is_empty() {
                anyhow::bail!("Search query cannot be empty");
            }
            models.retain(|m| m.id.to_lowercase().contains(&q));
            true
        } else {
            false
        };

        let label = if searching { "matches" } else { "models" };
        let provider_cell = if is_starter {
            let (plan_label, paid) = crate::commands::starter_provider_label(
                starter_account.as_ref().and_then(|a| a.plan.as_deref()),
                starter_account
                    .as_ref()
                    .and_then(|a| a.plan_label.as_deref()),
            );
            crate::commands::paint_plan_cell(paid, &plan_label)
        } else {
            style::dim(&key.base_url)
        };
        eprintln!(
            "{} {} {} via {}",
            style::success_symbol(),
            models.len(),
            label,
            provider_cell
        );

        if json {
            let mut payload = serde_json::json!({
                "provider": key.base_url,
                "models": models,
            });
            if let Some(plan) = starter_account.as_ref().and_then(|a| a.plan.as_deref()) {
                payload["plan"] = serde_json::json!(plan);
            }
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            let widths = ColumnWidths::from_models(&models);
            // Stdout's LineWriter flushes on every '\n', so a per-row println!
            // becomes one syscall per row. Batch into a single write_all.
            let mut buf = String::with_capacity(models.len() * 80);
            for model in &models {
                buf.push_str(&format_model_line(model, &widths));
                buf.push('\n');
            }
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            let _ = handle.write_all(buf.as_bytes());
        }

        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!("{} aivo models [options]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("List available models from the active API key's provider.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt("-k, --key <id|name>", "Select API key by ID or name");
        print_opt("-r, --refresh", "Bypass cache and fetch fresh model list");
        print_opt("-s, --search <query>", "Filter models by substring match");
        print_opt("--json", "Output model list as JSON instead of a table");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo models"));
        println!("  {}", style::dim("aivo models -s sonnet"));
        println!("  {}", style::dim("aivo models --json | jq '.models[].id'"));
    }
}

/// Returns just the scheme + host + port of a URL, e.g. "https://api.example.com".
/// Used to probe the root when the base URL includes a path segment like /endpoint.
fn url_origin(url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let mut origin = format!("{}://{}", parsed.scheme(), parsed.host_str()?);
    if let Some(port) = parsed.port() {
        origin.push_str(&format!(":{}", port));
    }
    Some(origin)
}

/// Returns true if the model is suitable for text chat. Single source of
/// truth lives in `services::model_compat::text_chat_incompat_reason`; this
/// is the boolean flip used by upstream filters (Copilot, hoisted
/// `chat_only` gate in `fetch_models_detailed_filtered`).
pub(crate) fn is_text_chat_model(id: &str) -> bool {
    crate::services::model_compat::text_chat_incompat_reason(id).is_none()
}

/// Copilot's Claude/OpenAI chat routing uses the chat completions API.
/// Exclude clearly responses-only Codex models that the endpoint rejects.
fn is_copilot_chat_model(id: &str) -> bool {
    is_text_chat_model(id) && !id.to_lowercase().contains("codex")
}

fn cloudflare_model_name(model: CloudflareModel) -> String {
    model
        .name
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(model.id)
}

/// Format a token count as a compact display string: 1M, 200K, 128K, etc.
fn format_token_count(n: u64) -> String {
    if n >= 1_000_000 {
        let m = n / 1_000_000;
        let remainder = (n % 1_000_000) / 100_000;
        if remainder > 0 {
            format!("{}.{}M", m, remainder)
        } else {
            format!("{}M", m)
        }
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}

/// Build a friendly error for a non-2xx response from a model-listing endpoint.
/// Leads with the user's intent ("No models found" for 404, "Could not fetch
/// models" otherwise), then explains why — extracting JSON `error.message`
/// when present and dropping HTML payloads. Replaces the raw multi-KB HTML
/// page that providers return for missing endpoints.
fn friendly_api_error(status: reqwest::StatusCode, body: &str) -> anyhow::Error {
    let lead = if status == reqwest::StatusCode::NOT_FOUND {
        "No models found"
    } else {
        "Could not fetch models"
    };
    let detail = extract_error_detail(body);
    let hint = status_hint(status);
    let reason = match (detail, hint) {
        (Some(d), Some(h)) => format!("{} ({})", d, h),
        (Some(d), None) => d,
        (None, Some(h)) => h.to_string(),
        (None, None) => format!("server returned {}", status),
    };
    anyhow::anyhow!("{} — {}", lead, reason)
}

/// Pull a one-line message out of an upstream error body. Recognizes common
/// JSON shapes (OpenAI/Anthropic `error.message`, Cloudflare `errors[0].message`,
/// generic `message`). Returns `None` for HTML payloads or empty bodies so the
/// caller can fall back to a status-only hint.
fn extract_error_detail(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        if let Some(msg) = value
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return Some(truncate_message(msg));
        }
        if let Some(msg) = value.get("error").and_then(|e| e.as_str()) {
            return Some(truncate_message(msg));
        }
        if let Some(msg) = value
            .get("errors")
            .and_then(|e| e.as_array())
            .and_then(|arr| arr.first())
            .and_then(|first| first.get("message"))
            .and_then(|m| m.as_str())
        {
            return Some(truncate_message(msg));
        }
        if let Some(msg) = value.get("message").and_then(|m| m.as_str()) {
            return Some(truncate_message(msg));
        }
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("<!doctype") || lower.starts_with("<html") || lower.starts_with("<?xml") {
        return None;
    }
    Some(truncate_message(trimmed))
}

fn truncate_message(s: &str) -> String {
    let s = s.trim();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i >= 200 {
            out.push('…');
            return out;
        }
        out.push(c);
    }
    out
}

fn status_hint(status: reqwest::StatusCode) -> Option<&'static str> {
    match status.as_u16() {
        401 => Some("authentication failed — check the API key with `aivo keys`"),
        403 => Some("this API key may not have permission to list models"),
        404 => {
            Some("the base URL may be wrong, or this provider doesn't expose a model listing API")
        }
        405 => Some("this provider does not expose a model listing API"),
        429 => Some("rate limited — try again shortly"),
        500..=599 => Some("the provider returned a server error — try again later"),
        _ => None,
    }
}

/// Convert a `/v1/models` price to a `$`-prefixed, 2-decimal per-1M display.
///
/// Providers report either per-token (OpenRouter — "0.000003") or per-million
/// (publicai — "0.15"). Per-token prices are always tiny, so any value >= 0.001
/// is already per-million and isn't scaled (else "0.15" became "$150000").
fn format_price_per_million(raw: &str) -> Option<String> {
    let price: f64 = raw.parse().ok()?;
    if price <= 0.0 {
        return None;
    }
    let per_m = if price >= 0.001 {
        price
    } else {
        price * 1_000_000.0
    };
    Some(format!("${per_m:.2}"))
}

#[derive(Default)]
struct ColumnWidths {
    name: usize,
    context: usize,
    max_output: usize,
}

impl ColumnWidths {
    fn from_models(models: &[ModelInfo]) -> Self {
        let mut w = Self::default();
        for m in models {
            w.name = w.name.max(m.id.len());
            if let Some(c) = &m.context {
                w.context = w.context.max(c.len());
            }
            if let Some(o) = &m.max_output {
                w.max_output = w.max_output.max(o.len());
            }
        }
        w
    }
}

fn format_model_line(model: &ModelInfo, widths: &ColumnWidths) -> String {
    let has_price = model.input_price.is_some() && model.output_price.is_some();
    let has_meta = model.context.is_some() || model.max_output.is_some() || has_price;
    if !has_meta && model.multiplier.is_none() && !model.deprecated {
        return model.id.clone();
    }

    let mut line = format!("{:<width$}", model.id, width = widths.name);
    let cw = widths.context.max(1);
    let ow = widths.max_output.max(1);

    // Compact ctx/out cell, e.g. "128K/16K"; a missing half is "-", a priced row
    // with no context is "-/-" so the price column still lines up.
    if has_meta {
        let ctx = model.context.as_deref().unwrap_or("-");
        let out = model.max_output.as_deref().unwrap_or("-");
        let cell = format!("{ctx}/{out}");
        // Pad to full width only when something follows, to avoid trailing space.
        let cell = if has_price || model.multiplier.is_some() || model.deprecated {
            format!("{cell:<width$}", width = cw + 1 + ow)
        } else {
            cell
        };
        line.push_str(&format!(" {}", style::dim(cell)));
    }

    // Left-aligned "$in/$out" per 1M — right-alignment leaves wide gaps ("$1.2/$6").
    if let (Some(input), Some(output)) = (&model.input_price, &model.output_price) {
        line.push_str(&format!(" {}", style::dim(format!("{input}/{output}"))));
    }
    if let Some(mult) = model.multiplier {
        line.push_str(&format!(" {}", style::dim(format_multiplier(mult))));
    }
    if model.deprecated {
        line.push_str(&format!(" {}", style::dim("deprecated")));
    }
    line
}

/// Build a compact picker label (id plus `Nx` suffix when a multiplier is known).
pub(crate) fn picker_label(model: &ModelInfo) -> String {
    match model.multiplier {
        Some(mult) => format!("{}  {}", model.id, format_multiplier(mult)),
        None => model.id.clone(),
    }
}

/// Format a Copilot premium-request multiplier as e.g. `1x`, `0.33x`, `7.5x`.
fn format_multiplier(m: f64) -> String {
    if m == m.trunc() {
        format!("{}x", m as i64)
    } else {
        let s = format!("{:.2}", m);
        format!("{}x", s.trim_end_matches('0').trim_end_matches('.'))
    }
}

pub(crate) async fn fetch_models(client: &Client, key: &ApiKey) -> Result<Vec<String>> {
    fetch_models_detailed(client, key)
        .await
        .map(|v| v.into_iter().map(|m| m.id).collect())
}

/// Fills missing context/max-output columns from the embedded models.dev
/// snapshot. Provider-reported values always win; never feed the result back
/// into the models cache.
fn enrich_from_snapshot(model: &mut ModelInfo) {
    let Some(snap) = crate::services::model_metadata::snapshot_limits(&model.id) else {
        return;
    };
    model.deprecated = snap.deprecated;
    if model.context_tokens.is_none()
        && let Some(context) = snap.context
    {
        model.context_tokens = Some(context);
        model.context = Some(format_token_count(context));
    }
    if model.max_output_tokens.is_none()
        && let Some(output) = snap.output
    {
        model.max_output_tokens = Some(output);
        model.max_output = Some(format_token_count(output));
    }
}

/// Pulls every displayed column out of a detailed model list so the cache can
/// reproduce the table on a warm read. Skips models that returned no
/// metadata at all (id-only entries from providers like Cloudflare).
fn build_metadata_map(models: &[ModelInfo]) -> HashMap<String, ModelMetadata> {
    models
        .iter()
        .filter_map(|m| {
            let meta = ModelMetadata {
                context_window: m.context_tokens,
                max_output: m.max_output.clone(),
                max_output_tokens: m.max_output_tokens,
                input_price: m.input_price.clone(),
                output_price: m.output_price.clone(),
                multiplier: m.multiplier,
                reasoning_efforts: m.reasoning_efforts.clone(),
            };
            let any = meta.context_window.is_some()
                || meta.max_output.is_some()
                || meta.input_price.is_some()
                || meta.output_price.is_some()
                || meta.multiplier.is_some()
                || !meta.reasoning_efforts.is_empty();
            any.then(|| (m.id.clone(), meta))
        })
        .collect()
}

pub(crate) fn model_cache_key_for_key(key: &ApiKey) -> String {
    if key.is_cursor_acp() {
        crate::services::cursor_acp::cursor_models_cache_identity(key)
    } else {
        key.base_url.clone()
    }
}

pub(crate) fn full_catalog_cache_key_for_key(key: &ApiKey) -> String {
    if key.is_cursor_acp() {
        // Cursor's `models` listing returns chat-capable ids only — there's no
        // image/audio/embedding catalog to separate. Collapse the `#all`
        // namespace into the bare key so `aivo models cursor`, the model
        // picker, and the in-router cache (see `cursor_bridge::cached_models`)
        // all hit a single shared entry on disk.
        return model_cache_key_for_key(key);
    }
    full_catalog_key(&model_cache_key_for_key(key))
}

/// Rebuilds the `aivo models` row list from a cached id list and metadata
/// map. Models present in `ids` but missing from `metadata` (e.g. Cloudflare
/// id-only entries) render as plain rows.
fn models_from_cache(ids: Vec<String>, metadata: HashMap<String, ModelMetadata>) -> Vec<ModelInfo> {
    ids.into_iter()
        .map(|id| {
            let m = metadata.get(&id).cloned().unwrap_or_default();
            ModelInfo {
                id,
                context: m.context_window.map(format_token_count),
                context_tokens: m.context_window,
                max_output: m.max_output,
                max_output_tokens: m.max_output_tokens,
                input_price: m.input_price,
                output_price: m.output_price,
                multiplier: m.multiplier,
                deprecated: false,
                reasoning_efforts: m.reasoning_efforts,
            }
        })
        .collect()
}

/// Best-effort warm of the per-model metadata cache (context window, pricing)
/// for a key's full catalog, so `model_metadata::resolve_limits` can answer
/// from cache without a picker/`aivo models` run first. Used by `aivo chat` to
/// populate the footer context-utilization stat on the `-m <model>` path, which
/// otherwise never fetches the catalog. Silently no-ops on fetch failure.
pub(crate) async fn warm_full_catalog_metadata(client: &Client, key: &ApiKey, cache: &ModelsCache) {
    if key.is_any_oauth() || crate::services::provider_profile::is_ollama_base(&key.base_url) {
        return;
    }
    let Ok(models) = fetch_models_detailed_filtered(client, key, false).await else {
        return;
    };
    let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
    let metadata = build_metadata_map(&models);
    cache
        .set_with_metadata(&full_catalog_cache_key_for_key(key), ids, metadata)
        .await;
}

/// Cache-first full catalog fetch; stores metadata so `resolve_limits` finds the
/// context window (else Pi/opencode fall back to a 128k default).
pub(crate) async fn fetch_all_models_cached(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
    bypass_cache: bool,
) -> Result<Vec<String>> {
    if !bypass_cache && let Some(cached) = full_catalog_cached(key, cache).await {
        return Ok(cached);
    }
    let detailed = fetch_models_detailed_filtered(client, key, false).await?;
    let models: Vec<String> = detailed.iter().map(|m| m.id.clone()).collect();
    if !crate::services::provider_profile::is_ollama_base(&key.base_url) {
        cache
            .set_with_metadata(
                &full_catalog_cache_key_for_key(key),
                models.clone(),
                build_metadata_map(&detailed),
            )
            .await;
    }
    Ok(models)
}

/// Peek the cached full catalog, honoring the same skips as
/// `fetch_all_models_cached`: Ollama never caches (lists locally, instantly),
/// and a cursor entry poisoned by the old parser bug counts as a miss.
async fn full_catalog_cached(key: &ApiKey, cache: &ModelsCache) -> Option<Vec<String>> {
    if crate::services::provider_profile::is_ollama_base(&key.base_url) {
        return None;
    }
    let cached = cache.get(&full_catalog_cache_key_for_key(key)).await?;
    (!cursor_cache_looks_corrupt(key, &cached)).then_some(cached)
}

/// Whether the key's cached catalog is still within its TTL. OAuth and Ollama
/// keys can't be harvested (`warm_full_catalog_metadata` no-ops), so they count
/// as fresh to skip a pointless warm.
pub(crate) async fn full_catalog_metadata_fresh(key: &ApiKey, cache: &ModelsCache) -> bool {
    if key.is_any_oauth() || crate::services::provider_profile::is_ollama_base(&key.base_url) {
        return true;
    }
    full_catalog_cached(key, cache).await.is_some()
}

/// Cache-first catalog fetch for the interactive model picker, shared by the
/// `run`/`start`/`chat` pickers. Returns instantly on a cache hit; shows a
/// "Fetching models…" spinner only while a genuine network fetch runs — a hit
/// must stay instant, since `stop_spinner` sleeps 100ms and would flash a frame.
/// Empty on fetch failure: callers treat that as "no list → use tool default".
pub(crate) async fn fetch_all_models_for_picker(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
    refresh: bool,
) -> Vec<String> {
    if !refresh && let Some(cached) = full_catalog_cached(key, cache).await {
        return cached;
    }
    let (spinning, handle) = style::start_spinner(Some(" Fetching models..."));
    let result = fetch_all_models_cached(client, key, cache, true).await;
    style::stop_spinner(&spinning);
    let _ = handle.await;
    result.unwrap_or_default()
}

/// Force-refetch when a cursor cache entry was populated by the older
/// parser bug — historical builds turned cursor-agent's logged-out
/// "No models available for this account." into the model id `No`, and
/// shipped that into `~/.config/aivo/models-cache.json`.
///
/// The earlier "must contain a digit / hyphen / slash" heuristic was too
/// broad: cursor's own catalog includes `auto` (no digit, no hyphen, no
/// slash), so every cache entry was treated as corrupt and refetched on
/// every call. Match the actual bad string instead.
fn cursor_cache_looks_corrupt(key: &ApiKey, cached: &[String]) -> bool {
    key.is_cursor_acp() && cached.iter().any(|id| id.trim() == "No")
}

/// Pure decision for `starter_model_still_available`: given a freshly-fetched
/// catalog, decide whether `model` is still listed. An empty catalog is
/// treated as a transient hiccup and passes through rather than flagging
/// every user's model as gone.
fn model_present_in_catalog(catalog: &[String], model: &str) -> bool {
    catalog.is_empty() || catalog.iter().any(|m| m == model)
}

/// Whether `model` is still listed for the aivo-starter server. `true` (skip)
/// for non-starter keys, the `aivo/starter` sentinel, `MODEL_DEFAULT_PLACEHOLDER`,
/// and an un-cached catalog.
///
/// Non-blocking by design: reads only the last-known catalog and never fetches,
/// so the chat/run/start launch never waits on `/v1/models` (that cost ~1s every
/// launch). Reads both starter cache spellings — the picker/`aivo models` key by
/// the sentinel, chat's background warm by the post-swap real URL — which are
/// never unified (the routers depend on the real-URL entry). The warm keeps it
/// fresh; a removal shows up a launch late, with the first request's error
/// bridging the gap.
pub(crate) async fn starter_model_still_available(
    key: &ApiKey,
    cache: &ModelsCache,
    model: &str,
) -> bool {
    if !is_aivo_starter_base(&key.base_url) {
        return true;
    }
    if model == crate::constants::AIVO_STARTER_MODEL
        || model == crate::constants::MODEL_DEFAULT_PLACEHOLDER
    {
        return true;
    }
    let mut catalog = cache
        .model_ids(crate::constants::AIVO_STARTER_SENTINEL)
        .await
        .unwrap_or_default();
    if let Some(more) = cache
        .model_ids(crate::constants::AIVO_STARTER_REAL_URL)
        .await
    {
        catalog.extend(more);
    }
    model_present_in_catalog(&catalog, model)
}

/// Fetch models with full metadata from the API where available.
/// Providers like OpenRouter/Vercel return context window, pricing, and max output.
/// Google returns inputTokenLimit and outputTokenLimit.
/// Other providers return just IDs.
pub(crate) async fn fetch_models_detailed(client: &Client, key: &ApiKey) -> Result<Vec<ModelInfo>> {
    fetch_models_detailed_filtered(client, key, true).await
}

/// Implementation behind `fetch_models_detailed` / `fetch_all_models_cached`. When
/// `chat_only` is true (the default), applies `is_text_chat_model` to the
/// OpenAI-compatible / Anthropic / CloudflareSearch branches so chat pickers
/// don't surface image, audio, or embedding models. Set to false for the
/// image command and `aivo models`, which show the full catalog.
async fn fetch_models_detailed_filtered(
    client: &Client,
    key: &ApiKey,
    chat_only: bool,
) -> Result<Vec<ModelInfo>> {
    if key.is_any_oauth() {
        anyhow::bail!(
            "Key '{}' is an OAuth credential with no model listing API. Use `{}` to launch directly, or switch to a regular API key with `aivo use`.",
            key.display_name(),
            key.oauth_tool_hint()
        );
    }
    let base = normalize_base_url(&key.base_url);
    let profile = provider_profile_for_key(key);

    let raw: Vec<ModelInfo> = match profile.model_listing_strategy {
        ModelListingStrategy::AivoStarter => {
            let starter_base = crate::constants::AIVO_STARTER_REAL_URL;
            let url = format!("{}/v1/models", starter_base.trim_end_matches('/'));
            let response = crate::services::device_fingerprint::with_starter_headers(
                client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", key.key.as_str())),
            )
            .send_logged()
            .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(friendly_api_error(status, &body));
            }

            let resp: OpenAIModelsResponse = response.json().await?;
            Ok::<_, anyhow::Error>(resp.data.into_iter().map(|m| m.into_model_info()).collect())
        }
        ModelListingStrategy::Ollama => {
            crate::services::ollama::ensure_ready().await?;
            let ids = crate::services::ollama::list_models().await?;
            Ok(ids.into_iter().map(ModelInfo::id_only).collect())
        }
        ModelListingStrategy::Copilot => {
            let tm = CopilotTokenManager::new(key.key.as_str().to_string());
            let (copilot_token, api_endpoint) = tm.get_token().await?;
            let url = format!("{}/models", api_endpoint.trim_end_matches('/'));
            let response = client
                .get(&url)
                .header("Authorization", format!("Bearer {}", copilot_token))
                .header("Editor-Version", COPILOT_EDITOR_VERSION)
                .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
                .header("X-GitHub-Api-Version", "2025-10-01")
                .send_logged()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(friendly_api_error(status, &body));
            }

            let resp: OpenAIModelsResponse = response.json().await?;
            Ok(resp
                .data
                .into_iter()
                .filter(|m| is_copilot_chat_model(&m.id))
                .map(|m| m.into_model_info())
                .collect())
        }
        ModelListingStrategy::CursorAcp => {
            let ids = crate::services::cursor_acp::list_cursor_models(key).await?;
            Ok(ids.into_iter().map(ModelInfo::id_only).collect())
        }
        ModelListingStrategy::Google => {
            let url = build_google_models_url(base);
            let response = client
                .get(&url)
                .header("x-goog-api-key", key.key.as_str())
                .send_logged()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(friendly_api_error(status, &body));
            }

            let resp: GeminiModelsResponse = response.json().await?;
            Ok(resp
                .models
                .into_iter()
                .filter(|m| {
                    m.supported_generation_methods
                        .iter()
                        .any(|method| method == "generateContent")
                })
                .map(|m| {
                    let id = m
                        .name
                        .strip_prefix("models/")
                        .unwrap_or(&m.name)
                        .to_string();
                    ModelInfo {
                        context: m.input_token_limit.map(format_token_count),
                        context_tokens: m.input_token_limit,
                        max_output: m.output_token_limit.map(format_token_count),
                        max_output_tokens: m.output_token_limit,
                        input_price: None,
                        output_price: None,
                        multiplier: None,
                        deprecated: false,
                        reasoning_efforts: Vec::new(),
                        id,
                    }
                })
                .collect())
        }
        ModelListingStrategy::Anthropic => {
            let url = build_anthropic_models_url(&key.base_url);
            let response = client
                .get(&url)
                .header("x-api-key", key.key.as_str())
                .header("anthropic-version", "2023-06-01")
                .send_logged()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(friendly_api_error(status, &body));
            }

            let resp: OpenAIModelsResponse = response.json().await?;
            Ok(resp.data.into_iter().map(|m| m.into_model_info()).collect())
        }
        ModelListingStrategy::CloudflareSearch => {
            let cloudflare_base = cloudflare_ai_base(base)
                .ok_or_else(|| anyhow::anyhow!("Failed to normalize Cloudflare AI base URL"))?;
            let auth = format!("Bearer {}", key.key.as_str());
            let mut page = 1u32;
            let mut seen = HashSet::new();
            let mut models = Vec::new();

            loop {
                let url = format!(
                    "{}/models/search?hide_experimental=true&page={}&per_page=100",
                    cloudflare_base, page
                );
                let response = client
                    .get(&url)
                    .header("Authorization", &auth)
                    .send_logged()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    return Err(friendly_api_error(status, &body));
                }

                let resp: CloudflareModelsResponse = response.json().await?;
                for model in resp.result.into_iter().map(cloudflare_model_name) {
                    if seen.insert(model.clone()) {
                        models.push(ModelInfo::id_only(model));
                    }
                }

                let total_pages = resp
                    .result_info
                    .and_then(|info| info.total_pages)
                    .unwrap_or(page);
                if page >= total_pages {
                    break;
                }
                page += 1;
            }

            Ok(models)
        }
        ModelListingStrategy::OpenAiCompatible => {
            let model_endpoints = |b: &str| [format!("{}/v1/models", b), format!("{}/models", b)];
            let mut candidates = Vec::new();
            if let Some(origin) = url_origin(base)
                && origin != base
            {
                candidates.extend(model_endpoints(&origin));
            }
            candidates.extend(model_endpoints(base));
            let auth = format!("Bearer {}", key.key.as_str());

            let mut last_err = String::new();
            let mut success: Option<Vec<ModelInfo>> = None;
            for url in &candidates {
                let response = client
                    .get(url)
                    .header("Authorization", &auth)
                    .send_logged()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    last_err = friendly_api_error(status, &body).to_string();
                    continue;
                }

                let body = response.text().await.unwrap_or_default();
                match serde_json::from_str::<OpenAIModelsResponse>(&body) {
                    Ok(resp) => {
                        success =
                            Some(resp.data.into_iter().map(|m| m.into_model_info()).collect());
                        break;
                    }
                    Err(e) => {
                        last_err = format!("Invalid models response from {}: {}", url, e);
                    }
                }
            }

            match success {
                Some(v) => Ok(v),
                None => anyhow::bail!("{}", last_err),
            }
        }
    }?;

    Ok(if chat_only {
        raw.into_iter()
            .filter(|m| is_text_chat_model(&m.id))
            .collect()
    } else {
        raw
    })
}

fn build_google_models_url(base_url: &str) -> String {
    let base = normalize_base_url(base_url).trim_end_matches('/');
    if base.ends_with("/v1beta") || base.ends_with("/v1") {
        format!("{}/models", base)
    } else if base.ends_with("/models") {
        base.to_string()
    } else {
        format!("{}/v1beta/models", base)
    }
}

fn build_anthropic_models_url(base_url: &str) -> String {
    http_utils::build_target_url(base_url, "/v1/models")
}

/// Fetches the model list (cache-first) with a spinner for network fetches,
/// filtered to text-chat models only. Used by chat and run commands for the
/// interactive model picker.
pub(crate) async fn fetch_models_for_select(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
) -> Vec<String> {
    fetch_models_cached(client, key, cache, false)
        .await
        .unwrap_or_default()
}

/// Cache-aware wrapper around `fetch_models`.
/// Returns cached result if present and not expired (unless `bypass_cache` is true).
/// On cache miss, fetches from the network and writes the result to the cache.
pub(crate) async fn fetch_models_cached(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
    bypass_cache: bool,
) -> Result<Vec<String>> {
    // Ollama lists local models instantly — skip cache entirely.
    let is_ollama = crate::services::provider_profile::is_ollama_base(&key.base_url);
    let cache_key = model_cache_key_for_key(key);
    if !bypass_cache
        && !is_ollama
        && let Some(cached) = cache.get(&cache_key).await
    {
        return Ok(cached);
    }
    let models = fetch_models(client, key).await?;
    if !is_ollama {
        cache.set(&cache_key, models.clone()).await;
    }
    Ok(models)
}

/// Determines whether the "(leave it to the tool)" default option should be
/// shown in the model picker for the given tool type and model list.
///
/// - Pi, Opencode: never (these tools require explicit model selection)
/// - Claude: only if the list contains Claude-family models
/// - Codex: only if the list contains gpt- prefixed models
/// - Gemini: only if the list contains Gemini-family models
pub(crate) fn tool_supports_default_model(tool: AIToolType, models: &[String]) -> bool {
    match tool {
        AIToolType::Pi | AIToolType::Opencode => false,
        AIToolType::Claude => models
            .iter()
            .any(|m| m.to_ascii_lowercase().contains("claude")),
        AIToolType::Codex | AIToolType::CodexApp => models
            .iter()
            .any(|m| m.to_ascii_lowercase().starts_with("gpt-")),
        AIToolType::Gemini => models
            .iter()
            .any(|m| m.to_ascii_lowercase().contains("gemini")),
    }
}

/// Shows an interactive model picker. The "(leave it to the tool)" default option
/// is conditionally shown based on whether the provider has models compatible with
/// the selected tool. When `tool` is `None` (e.g. chat mode), the default option
/// is hidden since a concrete model is required.
/// Returns `Some(MODEL_DEFAULT_PLACEHOLDER)` if the default is chosen,
/// `Some(model_name)` for a real model, or `None` if cancelled.
/// Outcome of picker-style model resolution. Distinguishes "user cancelled the
/// picker" (exit success, don't launch) from "no fetchable model list, fall back
/// to the tool's own default" (launch anyway, no injected model). Shared by
/// `aivo run` and the plugin endpoint so both resolve models identically.
pub(crate) enum ModelOutcome {
    /// User picked a model, or `--model <value>` was passed.
    Model(String),
    /// No `--model` flag, or the picker fetched an empty list. Launch with the
    /// tool's own default.
    UseDefault,
    /// Picker shown and cancelled (Ctrl-C / Esc) — caller should not launch.
    Cancelled,
}

/// Resolve the model to launch with. `--model <value>` → use as-is; bare
/// `--model` (`Some("")`) → fuzzy picker; **no `--model` (`None`) → `UseDefault`**
/// (let the tool use its own default — never a forced picker, so a bare launch
/// or a `-k`-only launch doesn't pop a dialog). On a non-TTY or an empty catalog
/// it also falls back to the default. `tool` colors the picker's "(leave it to
/// the tool)" row; pass `None` for a generic client (plugins).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn resolve_model_outcome(
    client: &Client,
    key: &ApiKey,
    flag_model: Option<String>,
    explicit_model_flag: bool,
    refresh: bool,
    tool: Option<AIToolType>,
    cache: &ModelsCache,
    prompt: &str,
) -> Result<ModelOutcome> {
    match flag_model {
        None => return Ok(ModelOutcome::UseDefault),
        Some(ref m) if !m.is_empty() => return Ok(ModelOutcome::Model(m.clone())),
        Some(_) => {}
    }

    // The picker raw-modes stdin and renders to stderr; without both TTYs it
    // can't run (a piped stdin reads as a cancel). Bail before the network fetch
    // so piped invocations don't pay for a catalog they can't show. Only explain
    // when the user explicitly asked for a picker.
    if !crate::tui::picker_interactive() {
        if explicit_model_flag {
            crate::commands::print_no_model_list_hint();
        }
        return Ok(ModelOutcome::UseDefault);
    }

    let models_list = fetch_all_models_for_picker(client, key, cache, refresh).await;
    if models_list.is_empty() {
        if explicit_model_flag {
            crate::commands::print_no_model_list_hint();
        }
        return Ok(ModelOutcome::UseDefault);
    }

    let annotations = crate::services::model_compat::text_chat_annotations(&models_list);
    match prompt_model_picker(models_list, tool, annotations, prompt) {
        Some(m) => Ok(ModelOutcome::Model(m)),
        None => Ok(ModelOutcome::Cancelled),
    }
}

/// Shows a fuzzy model picker with per-row annotations. `annotations[i] =
/// Some(reason)` disables the matching model and renders `reason` dim at the
/// end of the row (parallels the key picker). Pass `vec![]` for a vanilla
/// picker with every row selectable.
pub(crate) fn prompt_model_picker(
    models: Vec<String>,
    tool: Option<AIToolType>,
    annotations: Vec<Option<String>>,
    prompt: &str,
) -> Option<String> {
    use crate::constants;
    use crate::tui::FuzzySelect;

    let show_default = tool
        .map(|t| tool_supports_default_model(t, &models))
        .unwrap_or(false);

    let mut items = Vec::with_capacity(models.len() + show_default as usize);
    let mut row_annotations: Vec<Option<String>> = Vec::with_capacity(items.capacity());
    if show_default {
        items.push(constants::MODEL_DEFAULT_DISPLAY.to_string());
        row_annotations.push(None);
    }
    items.extend(models);
    // Align annotations with the model rows (after the optional default row).
    // A shorter/empty `annotations` means "no annotations" for those rows.
    for i in 0..items.len() - row_annotations.len() {
        row_annotations.push(annotations.get(i).cloned().flatten());
    }

    let selected = FuzzySelect::new()
        .with_prompt(prompt)
        .items(&items)
        .annotations(row_annotations)
        .default(0)
        .interact_opt()
        .ok()
        .flatten()?;

    if show_default && selected == 0 {
        Some(constants::MODEL_DEFAULT_PLACEHOLDER.to_string())
    } else {
        Some(items[selected].clone())
    }
}

/// Converts the `__default__` placeholder to `None` for passing to tools.
pub(crate) fn resolve_model_placeholder(model: Option<String>) -> Option<String> {
    match model.as_deref() {
        Some(crate::constants::MODEL_DEFAULT_PLACEHOLDER) => None,
        _ => model,
    }
}

/// Returns a display-friendly model string, converting `__default__` and `None` to "(tool default)".
pub(crate) fn model_display_label(model: Option<&str>) -> &str {
    match model {
        Some(crate::constants::MODEL_DEFAULT_PLACEHOLDER) | None => "(tool default)",
        Some(m) => m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::models_cache::ModelsCache;
    use tempfile::TempDir;

    fn make_key(url: &str) -> ApiKey {
        use zeroize::Zeroizing;
        ApiKey {
            id: "1".to_string(),
            name: "test".to_string(),
            base_url: url.to_string(),
            claude_protocol: None,
            gemini_protocol: None,
            responses_api_supported: None,
            codex_mode: None,
            opencode_mode: None,
            pi_mode: None,
            claude_path_variant: None,
            gemini_path_variant: None,
            requires_reasoning_content: None,
            protocol_routes: Default::default(),
            routing_schema_version: 0,
            key: Zeroizing::new("sk-test".to_string()),
            created_at: "2026-01-01".to_string(),
        }
    }

    #[test]
    fn cursor_cache_corrupt_predicate_rejects_logged_out_marker_only() {
        // Regression: the previous heuristic ("must contain digit / hyphen /
        // slash") rejected every cursor cache entry because `auto` is a real
        // cursor model id with none of those characters. That defeated the
        // disk cache entirely — every `aivo models` and every cursor router
        // /v1/models hit refetched from cursor-agent.
        let mut cursor_key = make_key(crate::services::cursor_acp::CURSOR_ACP_SENTINEL);
        cursor_key.key = zeroize::Zeroizing::new(format!(
            "{}testaccount1",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        let real_models = vec![
            "auto".to_string(),
            "composer-2.5".to_string(),
            "claude-sonnet-4-6".to_string(),
        ];
        assert!(
            !cursor_cache_looks_corrupt(&cursor_key, &real_models),
            "`auto` is a real cursor model and must not flag the cache as corrupt"
        );
        // The original sentinel that the predicate exists to catch.
        let bad_models = vec!["No".to_string()];
        assert!(cursor_cache_looks_corrupt(&cursor_key, &bad_models));
        // Non-cursor keys are unaffected.
        let other = make_key("https://api.deepseek.com");
        assert!(!cursor_cache_looks_corrupt(&other, &bad_models));
    }

    #[test]
    fn full_catalog_cache_key_collapses_cursor_into_shared_namespace() {
        // Cursor listings are chat-only, so `aivo models cursor` and the
        // picker must hit a single cache entry. Other providers keep the
        // `#all` namespace to separate the broad catalog from the chat
        // picker's filtered list.
        let mut cursor_key = make_key(crate::services::cursor_acp::CURSOR_ACP_SENTINEL);
        cursor_key.key = zeroize::Zeroizing::new(format!(
            "{}testaccount1",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        assert_eq!(
            full_catalog_cache_key_for_key(&cursor_key),
            model_cache_key_for_key(&cursor_key),
        );
        // Non-cursor keys retain the `#all` suffix so chat pickers don't
        // inherit image/audio/embedding entries from the broad fetch.
        let other = make_key("https://api.deepseek.com");
        assert_ne!(
            full_catalog_cache_key_for_key(&other),
            model_cache_key_for_key(&other),
        );
        assert!(full_catalog_cache_key_for_key(&other).ends_with("#all"));
    }

    #[tokio::test]
    async fn full_catalog_metadata_fresh_tracks_cache_ttl() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = make_key("https://api.getaivo.dev");

        // No entry yet → stale.
        assert!(!full_catalog_metadata_fresh(&key, &cache).await);

        // Just written → fresh.
        cache
            .set(
                &full_catalog_cache_key_for_key(&key),
                vec!["aivo/starter".to_string()],
            )
            .await;
        assert!(full_catalog_metadata_fresh(&key, &cache).await);
    }

    #[tokio::test]
    async fn full_catalog_metadata_fresh_reports_expired_entry_as_stale() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models-cache.json");
        // fetched_at = 0 is past the TTL.
        let key = make_key("https://api.getaivo.dev");
        let entry = serde_json::json!({
            full_catalog_cache_key_for_key(&key): {
                "models": ["aivo/starter"],
                "fetched_at": 0u64
            }
        });
        tokio::fs::write(&path, serde_json::to_string(&entry).unwrap())
            .await
            .unwrap();
        let cache = ModelsCache::with_path(path);
        assert!(!full_catalog_metadata_fresh(&key, &cache).await);
    }

    #[test]
    fn test_is_text_chat_model_keeps_chat_models() {
        assert!(is_text_chat_model("gpt-4o"));
        assert!(is_text_chat_model("gpt-4o-mini"));
        assert!(is_text_chat_model("claude-sonnet-4-6"));
        assert!(is_text_chat_model("gpt-3.5-turbo"));
        assert!(is_text_chat_model("o1"));
        assert!(is_text_chat_model("o3-mini"));
        assert!(is_text_chat_model("gpt-4o-audio-preview"));
        assert!(is_text_chat_model("gemini-1.5-pro"));
        assert!(is_text_chat_model("gemini-2.0-flash"));
    }

    #[test]
    fn test_is_text_chat_model_filters_embeddings() {
        assert!(!is_text_chat_model("text-embedding-3-small"));
        assert!(!is_text_chat_model("text-embedding-3-large"));
        assert!(!is_text_chat_model("text-embedding-ada-002"));
        assert!(!is_text_chat_model("embedding-001"));
        assert!(!is_text_chat_model("text-embeddings-inference"));
    }

    #[test]
    fn test_is_text_chat_model_filters_image_and_audio() {
        assert!(!is_text_chat_model("dall-e-2"));
        assert!(!is_text_chat_model("dall-e-3"));
        assert!(!is_text_chat_model("tts-1"));
        assert!(!is_text_chat_model("tts-1-hd"));
        assert!(!is_text_chat_model("whisper-1"));
        assert!(!is_text_chat_model("gpt-image-1"));
        assert!(!is_text_chat_model("google/gemini-3.1-flash-image-preview"));
    }

    #[test]
    fn test_is_copilot_chat_model_filters_codex_models() {
        assert!(is_copilot_chat_model("gpt-4o"));
        assert!(is_copilot_chat_model("claude-sonnet-4"));
        assert!(!is_copilot_chat_model("gpt-5.1-codex-mini"));
        assert!(!is_copilot_chat_model("gpt-5.3-codex"));
        assert!(!is_copilot_chat_model("openai/gpt-5.1-codex-mini"));
    }

    #[test]
    fn cloudflare_ai_base_normalizes_v1_suffix() {
        assert_eq!(
            cloudflare_ai_base("https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"),
            Some("https://api.cloudflare.com/client/v4/accounts/abc/ai".to_string())
        );
    }

    #[test]
    fn cloudflare_ai_base_accepts_ai_root() {
        assert_eq!(
            cloudflare_ai_base("https://api.cloudflare.com/client/v4/accounts/abc/ai"),
            Some("https://api.cloudflare.com/client/v4/accounts/abc/ai".to_string())
        );
    }

    #[test]
    fn cloudflare_ai_base_rejects_non_cloudflare() {
        assert_eq!(cloudflare_ai_base("https://api.openai.com/v1"), None);
    }

    #[test]
    fn cloudflare_model_name_prefers_name_over_id() {
        let model: CloudflareModel = serde_json::from_str(
            r#"{"id":"01564c52-8717-47dc-8efd-907a2ca18301","name":"@cf/meta/llama-3.1-8b-instruct"}"#,
        )
        .unwrap();
        assert_eq!(
            cloudflare_model_name(model),
            "@cf/meta/llama-3.1-8b-instruct".to_string()
        );
    }

    #[test]
    fn cloudflare_model_name_falls_back_to_id() {
        let model: CloudflareModel =
            serde_json::from_str(r#"{"id":"01564c52-8717-47dc-8efd-907a2ca18301"}"#).unwrap();
        assert_eq!(
            cloudflare_model_name(model),
            "01564c52-8717-47dc-8efd-907a2ca18301".to_string()
        );
    }

    #[test]
    fn build_anthropic_models_url_preserves_v1_path() {
        assert_eq!(
            build_anthropic_models_url("https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/models"
        );
        assert_eq!(
            build_anthropic_models_url("https://api.minimax.io/anthropic"),
            "https://api.minimax.io/anthropic/v1/models"
        );
    }

    #[tokio::test]
    async fn cached_models_returned_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let models = vec!["model-a".to_string()];
        cache.set("https://api.example.com", models.clone()).await;

        let key = make_key("https://api.example.com");
        let client = reqwest::Client::new();
        // With a valid cache, fetch_models_cached should return cached list
        // without making a network call (network call would fail with this fake key)
        let result = fetch_models_cached(&client, &key, &cache, false).await;
        assert_eq!(result.unwrap(), models);
    }

    #[tokio::test]
    async fn bypass_cache_ignores_warm_cache() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        // Seed cache with stale data
        cache
            .set("https://api.example.com", vec!["stale-model".to_string()])
            .await;

        let key = make_key("https://api.example.com");
        let client = reqwest::Client::new();
        // With bypass_cache=true, the function should NOT return the cached value.
        // It will try a network call (which will fail with a fake key) — that's fine,
        // we just verify it didn't return the cached stale data.
        let result = fetch_models_cached(&client, &key, &cache, true).await;
        // Network call will fail (fake key) — result should be Err, not the stale cached value
        assert!(
            result.is_err(),
            "Expected network error, not cached stale data"
        );
    }

    // Tests for `starter_model_still_available`. Short-circuit paths (no
    // network) are exercised through the full helper; catalog-content
    // decisions are tested through the pure `model_present_in_catalog` so
    // they don't depend on HTTP fixtures.

    #[tokio::test]
    async fn starter_validation_passes_non_starter_keys_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = make_key("https://api.example.com");
        assert!(starter_model_still_available(&key, &cache, "any-model").await);
    }

    #[tokio::test]
    async fn starter_validation_reads_cached_catalog_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = make_key(crate::constants::AIVO_STARTER_SENTINEL);

        // Empty cache → can't judge → passes (never blocks a fresh install).
        assert!(starter_model_still_available(&key, &cache, "model-a").await);

        // The chat warm keys the catalog by the post-swap real URL; validation
        // must read it even though the key's base_url is the sentinel.
        cache
            .set(
                &full_catalog_key(crate::constants::AIVO_STARTER_REAL_URL),
                vec!["model-a".to_string()],
            )
            .await;
        assert!(starter_model_still_available(&key, &cache, "model-a").await);
        assert!(!starter_model_still_available(&key, &cache, "model-gone").await);
    }

    #[tokio::test]
    async fn starter_validation_passes_sentinel_model_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = make_key(crate::constants::AIVO_STARTER_SENTINEL);
        // `aivo/starter` is the stable sentinel — the helper must short-circuit.
        assert!(
            starter_model_still_available(&key, &cache, crate::constants::AIVO_STARTER_MODEL,)
                .await
        );
    }

    #[tokio::test]
    async fn starter_validation_passes_default_placeholder_without_network() {
        let dir = TempDir::new().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = make_key(crate::constants::AIVO_STARTER_SENTINEL);
        assert!(
            starter_model_still_available(
                &key,
                &cache,
                crate::constants::MODEL_DEFAULT_PLACEHOLDER,
            )
            .await
        );
    }

    #[test]
    fn model_present_when_catalog_contains_it() {
        let catalog = vec!["a".to_string(), "b".to_string()];
        assert!(model_present_in_catalog(&catalog, "a"));
        assert!(model_present_in_catalog(&catalog, "b"));
    }

    #[test]
    fn model_absent_when_catalog_does_not_contain_it() {
        let catalog = vec!["a".to_string(), "b".to_string()];
        assert!(!model_present_in_catalog(&catalog, "removed"));
    }

    #[test]
    fn empty_catalog_passes_through() {
        // A `200 OK` with `data: []` is almost always a transient server bug,
        // not "every model was removed". Pass through rather than flagging
        // every user's persisted model as gone.
        assert!(model_present_in_catalog(&[], "anything"));
    }

    #[test]
    fn test_tool_supports_default_pi_always_false() {
        let models = vec!["claude-sonnet-4-6".into(), "gpt-4o".into()];
        assert!(!tool_supports_default_model(AIToolType::Pi, &models));
    }

    #[test]
    fn test_tool_supports_default_opencode_always_false() {
        let models = vec!["claude-sonnet-4-6".into(), "gpt-4o".into()];
        assert!(!tool_supports_default_model(AIToolType::Opencode, &models));
    }

    #[test]
    fn test_tool_supports_default_claude_with_claude_models() {
        let models = vec!["claude-sonnet-4-6".into(), "gpt-4o".into()];
        assert!(tool_supports_default_model(AIToolType::Claude, &models));
    }

    #[test]
    fn test_tool_supports_default_claude_without_claude_models() {
        let models = vec!["gpt-4o".into(), "gemini-2.5-pro".into()];
        assert!(!tool_supports_default_model(AIToolType::Claude, &models));
    }

    #[test]
    fn test_tool_supports_default_codex_with_gpt_models() {
        let models = vec!["gpt-4o".into(), "claude-sonnet-4-6".into()];
        assert!(tool_supports_default_model(AIToolType::Codex, &models));
    }

    #[test]
    fn test_tool_supports_default_codex_only_matches_gpt_prefix() {
        // o-series and chatgpt should NOT match for Codex
        let models = vec!["o3-mini".into(), "o4-preview".into(), "chatgpt-4o".into()];
        assert!(!tool_supports_default_model(AIToolType::Codex, &models));
    }

    #[test]
    fn test_tool_supports_default_codex_without_gpt_models() {
        let models = vec!["claude-sonnet-4-6".into(), "gemini-2.5-pro".into()];
        assert!(!tool_supports_default_model(AIToolType::Codex, &models));
    }

    #[test]
    fn test_tool_supports_default_gemini_with_gemini_models() {
        let models = vec!["gemini-2.5-pro".into(), "gpt-4o".into()];
        assert!(tool_supports_default_model(AIToolType::Gemini, &models));
    }

    #[test]
    fn test_tool_supports_default_gemini_without_gemini_models() {
        let models = vec!["gpt-4o".into(), "claude-sonnet-4-6".into()];
        assert!(!tool_supports_default_model(AIToolType::Gemini, &models));
    }

    #[test]
    fn extract_error_detail_openai_shape() {
        let body = r#"{"error":{"message":"Invalid API key","type":"invalid_request_error"}}"#;
        assert_eq!(
            extract_error_detail(body).as_deref(),
            Some("Invalid API key")
        );
    }

    #[test]
    fn extract_error_detail_cloudflare_shape() {
        let body = r#"{"success":false,"errors":[{"code":7000,"message":"No route for the URI"}],"messages":[],"result":null}"#;
        assert_eq!(
            extract_error_detail(body).as_deref(),
            Some("No route for the URI")
        );
    }

    #[test]
    fn extract_error_detail_simple_error_string() {
        let body = r#"{"error":"unauthorized"}"#;
        assert_eq!(extract_error_detail(body).as_deref(), Some("unauthorized"));
    }

    #[test]
    fn extract_error_detail_drops_html() {
        let body = "<!DOCTYPE html><html><body><h1>404 Not Found</h1></body></html>";
        assert!(extract_error_detail(body).is_none());
        let body = "<html><head></head><body>oops</body></html>";
        assert!(extract_error_detail(body).is_none());
    }

    #[test]
    fn extract_error_detail_handles_empty() {
        assert!(extract_error_detail("").is_none());
        assert!(extract_error_detail("   \n\t  ").is_none());
    }

    #[test]
    fn extract_error_detail_truncates_plain_text() {
        let body = "x".repeat(500);
        let out = extract_error_detail(&body).unwrap();
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), 201);
    }

    #[test]
    fn status_hint_known_codes() {
        assert!(
            status_hint(reqwest::StatusCode::UNAUTHORIZED)
                .unwrap()
                .contains("API key")
        );
        assert!(
            status_hint(reqwest::StatusCode::FORBIDDEN)
                .unwrap()
                .contains("permission")
        );
        assert!(
            status_hint(reqwest::StatusCode::NOT_FOUND)
                .unwrap()
                .contains("base URL")
        );
        assert!(
            status_hint(reqwest::StatusCode::TOO_MANY_REQUESTS)
                .unwrap()
                .contains("rate")
        );
        assert!(
            status_hint(reqwest::StatusCode::INTERNAL_SERVER_ERROR)
                .unwrap()
                .contains("server")
        );
        assert!(status_hint(reqwest::StatusCode::IM_A_TEAPOT).is_none());
    }

    #[test]
    fn friendly_api_error_404_leads_with_no_models_found() {
        let err = friendly_api_error(
            reqwest::StatusCode::NOT_FOUND,
            "<!doctype html><html><body>404 page</body></html>",
        );
        let s = err.to_string();
        assert!(s.starts_with("No models found"));
        assert!(s.contains("base URL may be wrong"));
        assert!(!s.contains("<html"));
        assert!(!s.contains("<body"));
        assert!(!s.contains("404"));
    }

    #[test]
    fn friendly_api_error_401_includes_extracted_message() {
        let err = friendly_api_error(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"Invalid API key"}}"#,
        );
        let s = err.to_string();
        assert!(s.starts_with("Could not fetch models"));
        assert!(s.contains("Invalid API key"));
        assert!(s.contains("aivo keys"));
        assert!(!s.contains("401"));
    }

    #[test]
    fn friendly_api_error_unknown_status_falls_back_to_status_text() {
        let err = friendly_api_error(reqwest::StatusCode::IM_A_TEAPOT, "");
        let s = err.to_string();
        assert!(s.starts_with("Could not fetch models"));
        assert!(s.contains("server returned 418"));
    }

    #[test]
    fn friendly_api_error_500_uses_server_error_hint() {
        let err = friendly_api_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "");
        let s = err.to_string();
        assert!(s.starts_with("Could not fetch models"));
        assert!(s.contains("server error"));
    }

    #[test]
    fn models_from_cache_round_trips_full_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "openrouter/sonnet".to_string(),
            ModelMetadata {
                context_window: Some(200_000),
                max_output: Some("32K".to_string()),
                max_output_tokens: Some(32_000),
                input_price: Some("$3".to_string()),
                output_price: Some("$15".to_string()),
                multiplier: None,
                reasoning_efforts: Vec::new(),
            },
        );
        metadata.insert(
            "copilot/gpt-5".to_string(),
            ModelMetadata {
                multiplier: Some(0.5),
                ..Default::default()
            },
        );

        let ids = vec![
            "openrouter/sonnet".to_string(),
            "copilot/gpt-5".to_string(),
            "id-only-model".to_string(),
        ];
        let models = models_from_cache(ids, metadata);

        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "openrouter/sonnet");
        assert_eq!(models[0].context.as_deref(), Some("200K"));
        assert_eq!(models[0].context_tokens, Some(200_000));
        assert_eq!(models[0].max_output.as_deref(), Some("32K"));
        assert_eq!(models[0].input_price.as_deref(), Some("$3"));
        assert_eq!(models[0].output_price.as_deref(), Some("$15"));

        assert_eq!(models[1].multiplier, Some(0.5));
        assert!(models[1].context.is_none());

        // Models without metadata reconstruct as id-only rows.
        assert_eq!(models[2].id, "id-only-model");
        assert!(models[2].context.is_none());
        assert!(models[2].input_price.is_none());
    }

    #[test]
    fn build_metadata_map_skips_id_only_models() {
        let models = vec![
            ModelInfo {
                id: "rich".to_string(),
                context: Some("128K".to_string()),
                context_tokens: Some(128_000),
                max_output: Some("8K".to_string()),
                max_output_tokens: Some(8_000),
                input_price: Some("$1".to_string()),
                output_price: Some("$2".to_string()),
                multiplier: None,
                deprecated: false,
                reasoning_efforts: Vec::new(),
            },
            ModelInfo::id_only("bare".to_string()),
        ];
        let map = build_metadata_map(&models);
        assert!(map.contains_key("rich"));
        assert!(!map.contains_key("bare"));
        let rich = map.get("rich").unwrap();
        assert_eq!(rich.context_window, Some(128_000));
        assert_eq!(rich.max_output.as_deref(), Some("8K"));
    }

    #[test]
    fn cursor_cache_key_separates_shadow_accounts_and_api_keys() {
        let mut shadow_a = make_key(crate::services::cursor_acp::CURSOR_ACP_SENTINEL);
        shadow_a.key = zeroize::Zeroizing::new(format!(
            "{}aaaa1111",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        let mut shadow_b = make_key(crate::services::cursor_acp::CURSOR_ACP_SENTINEL);
        shadow_b.key = zeroize::Zeroizing::new(format!(
            "{}bbbb2222",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        let a = model_cache_key_for_key(&shadow_a);
        let b = model_cache_key_for_key(&shadow_b);
        assert!(a.starts_with("cursor#shadow-"));
        assert_ne!(a, b);

        let mut api = make_key(crate::services::cursor_acp::CURSOR_ACP_SENTINEL);
        api.key = zeroize::Zeroizing::new("sk-cursor".to_string());
        let cache_key = model_cache_key_for_key(&api);
        assert!(cache_key.starts_with("cursor#"));
        assert!(!cache_key.starts_with("cursor#shadow-"));
        assert!(!cache_key.contains("sk-cursor"));
    }

    #[test]
    fn test_tool_supports_default_empty_model_list() {
        assert!(!tool_supports_default_model(AIToolType::Claude, &[]));
        assert!(!tool_supports_default_model(AIToolType::Codex, &[]));
        assert!(!tool_supports_default_model(AIToolType::Gemini, &[]));
        assert!(!tool_supports_default_model(AIToolType::Pi, &[]));
        assert!(!tool_supports_default_model(AIToolType::Opencode, &[]));
    }

    #[test]
    fn format_price_scales_per_token_values() {
        // OpenRouter/Vercel report per-token (tiny) values; always 2 decimals.
        assert_eq!(
            format_price_per_million("0.0000005").as_deref(),
            Some("$0.50")
        );
        assert_eq!(
            format_price_per_million("0.000003").as_deref(),
            Some("$3.00")
        );
        // Even the priciest realistic model ($200/1M) stays per-token.
        assert_eq!(
            format_price_per_million("0.0002").as_deref(),
            Some("$200.00")
        );
    }

    #[test]
    fn format_price_keeps_per_million_values() {
        // publicai reports per-million dollars directly — must not be scaled.
        assert_eq!(format_price_per_million("0.15").as_deref(), Some("$0.15"));
        assert_eq!(format_price_per_million("0.2").as_deref(), Some("$0.20"));
        assert_eq!(format_price_per_million("2.92").as_deref(), Some("$2.92"));
        assert_eq!(format_price_per_million("200").as_deref(), Some("$200.00"));
    }

    #[test]
    fn format_price_rejects_zero_and_garbage() {
        assert_eq!(format_price_per_million("0"), None);
        assert_eq!(format_price_per_million("-1"), None);
        assert_eq!(format_price_per_million("free"), None);
    }

    #[test]
    fn format_model_line_shows_price_without_context() {
        // A price-only model (no ctx/max_output) still renders its price.
        let mut m = ModelInfo::id_only("allenai/Olmo-3.1-32B-Think".to_string());
        m.input_price = Some("$0.05".to_string());
        m.output_price = Some("$0.20".to_string());
        let widths = ColumnWidths::from_models(std::slice::from_ref(&m));
        let line = format_model_line(&m, &widths);
        assert!(line.contains("$0.05/$0.20"), "price missing: {line:?}");
    }

    #[test]
    fn format_model_line_bare_id_when_no_info() {
        let m = ModelInfo::id_only("some/model".to_string());
        let widths = ColumnWidths::from_models(std::slice::from_ref(&m));
        assert_eq!(format_model_line(&m, &widths), "some/model");
    }

    #[test]
    fn format_model_line_dashes_missing_meta_values() {
        // A missing context/max-output half renders as "-" in the compact cell.
        let mut ctx_only = ModelInfo::id_only("m".to_string());
        ctx_only.context = Some("512".to_string());
        let widths = ColumnWidths::from_models(std::slice::from_ref(&ctx_only));
        let line = console::strip_ansi_codes(&format_model_line(&ctx_only, &widths)).into_owned();
        assert!(line.contains("512/-"), "expected '512/-', got {line:?}");

        // A priced row with no context shows "-/-" so the price column aligns.
        let mut price_only = ModelInfo::id_only("m".to_string());
        price_only.input_price = Some("$0.10".to_string());
        price_only.output_price = Some("$0.20".to_string());
        let widths = ColumnWidths::from_models(std::slice::from_ref(&price_only));
        let line = console::strip_ansi_codes(&format_model_line(&price_only, &widths)).into_owned();
        assert!(line.contains("-/-"), "expected '-/-', got {line:?}");
        assert!(line.contains("$0.10/$0.20"), "price missing: {line:?}");
    }

    #[test]
    fn format_model_line_price_column_aligns_with_and_without_context() {
        // The price column starts at the same offset with or without a context.
        let mut with_ctx = ModelInfo::id_only("aaaa".to_string());
        with_ctx.context = Some("128K".to_string());
        with_ctx.max_output = Some("8K".to_string());
        with_ctx.input_price = Some("$0.15".to_string());
        with_ctx.output_price = Some("$0.20".to_string());
        let mut no_ctx = ModelInfo::id_only("b".to_string());
        no_ctx.input_price = Some("$0.05".to_string());
        no_ctx.output_price = Some("$0.20".to_string());

        let models = [with_ctx, no_ctx];
        let widths = ColumnWidths::from_models(&models);
        let l0 = console::strip_ansi_codes(&format_model_line(&models[0], &widths)).into_owned();
        let l1 = console::strip_ansi_codes(&format_model_line(&models[1], &widths)).into_owned();

        let col = |line: &str, needle: &str| line.find(needle).map(|b| line[..b].chars().count());
        assert_eq!(
            col(&l0, "0.15"),
            col(&l1, "0.05"),
            "price columns misaligned:\n{l0:?}\n{l1:?}"
        );
    }
}
