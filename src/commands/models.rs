/**
 * ModelsCommand handler for listing available models from the active provider.
 * Calls provider-specific model listing endpoints (OpenAI, Gemini, Cloudflare).
 */
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
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
    ModelListingStrategy, cloudflare_ai_base, is_aivo_starter_base, provider_profile_for_base_url,
    provider_profile_for_key,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiplier: Option<f64>,
}

impl ModelInfo {
    fn id_only(id: String) -> Self {
        Self {
            id,
            context: None,
            context_tokens: None,
            max_output: None,
            input_price: None,
            output_price: None,
            multiplier: None,
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
        let max_output = self.find_max_output().map(format_token_count);
        let (input_price, output_price) = self.find_pricing();
        let multiplier = self.find_multiplier();
        ModelInfo {
            id: self.id,
            context,
            context_tokens,
            max_output,
            input_price,
            output_price,
            multiplier,
        }
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
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
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
        let key = match key_override {
            Some(k) => k,
            None => match self.session_store.get_active_key().await? {
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

        let profile = provider_profile_for_base_url(&key.base_url);
        let is_static = matches!(
            profile.model_listing_strategy,
            ModelListingStrategy::Static(_)
        );

        let client = http_utils::router_http_client();
        let is_ollama = crate::services::provider_profile::is_ollama_base(&key.base_url);
        let all_cache_key = full_catalog_key(&key.base_url);
        let cache_warm = !refresh && self.cache.get(&all_cache_key).await.is_some();

        // `aivo models` shows the provider's full catalog, including image,
        // audio, and embedding models. Chat pickers filter/annotate at their
        // own call sites.
        let mut models = if cache_warm {
            fetch_models_detailed_filtered(&client, &key, false).await?
        } else {
            let started_at = Instant::now();
            let (spinning, spinner_handle) = style::start_spinner(Some(" Fetching models..."));
            let result = fetch_models_detailed_filtered(&client, &key, false).await;
            let min_visible = Duration::from_millis(350);
            if let Some(remaining) = min_visible.checked_sub(started_at.elapsed()) {
                tokio::time::sleep(remaining).await;
            }
            style::stop_spinner(&spinning);
            let _ = spinner_handle.await;
            result?
        };

        // Cache the full list (including image/audio/embed) under the `#all`
        // namespace so `aivo image` and future broad pickers can share it.
        // Also persist per-model context-window metadata so `aivo run claude`
        // can default `--max-context` for known 1M/2M models without making
        // its own network call.
        if !is_ollama {
            let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
            let metadata = build_metadata_map(&models);
            self.cache
                .set_with_metadata(&all_cache_key, ids, metadata)
                .await;
        }

        let is_starter = is_aivo_starter_base(&key.base_url);
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
        eprintln!(
            "{} {} {} via {}",
            style::success_symbol(),
            models.len(),
            label,
            style::dim(&key.base_url)
        );

        if json {
            let payload = serde_json::json!({
                "provider": key.base_url,
                "is_static": is_static,
                "models": models,
            });
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            let widths = ColumnWidths::from_models(&models);
            for model in &models {
                println!("{}", format_model_line(model, &widths));
            }

            if is_static {
                eprintln!(
                    "{}",
                    style::dim(
                        "Note: This provider does not have a model listing API. Showing a built-in list."
                    )
                );
            }
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
        println!(
            "{}",
            style::dim(
                "Calls /v1/models (OpenAI/Anthropic-compatible), /v1beta/models (Google), or /ai/models/search (Cloudflare)."
            )
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
        print_opt(
            "-k, --key <id|name>",
            "Select API key by ID or name (-k opens key picker)",
        );
        print_opt("-r, --refresh", "Bypass cache and fetch fresh model list");
        print_opt("-s, --search <query>", "Filter models by substring match");
        print_opt("--json", "Output model list as JSON instead of a table");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo models"));
        println!("  {}", style::dim("aivo models -s sonnet"));
        println!("  {}", style::dim("aivo models --key openrouter"));
        println!("  {}", style::dim("aivo models --refresh"));
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

/// Convert a per-token price string (e.g. "0.000003") to per-1M display (e.g. "$3").
fn format_price_per_million(per_token: &str) -> Option<String> {
    let price: f64 = per_token.parse().ok()?;
    if price <= 0.0 {
        return None;
    }
    let per_m = price * 1_000_000.0;
    if per_m >= 1.0 {
        let rounded = (per_m * 100.0).round() / 100.0;
        if rounded == rounded.floor() {
            Some(format!("${}", rounded as u64))
        } else {
            let s = format!("{:.2}", rounded);
            Some(format!(
                "${}",
                s.trim_end_matches('0').trim_end_matches('.')
            ))
        }
    } else {
        let s = format!("{:.4}", per_m);
        Some(format!(
            "${}",
            s.trim_end_matches('0').trim_end_matches('.')
        ))
    }
}

#[derive(Default)]
struct ColumnWidths {
    name: usize,
    context: usize,
    max_output: usize,
    input_price: usize,
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
            if let Some(p) = &m.input_price {
                w.input_price = w.input_price.max(p.len());
            }
        }
        w
    }
}

fn format_model_line(model: &ModelInfo, widths: &ColumnWidths) -> String {
    let has_info =
        model.context.is_some() || model.max_output.is_some() || model.multiplier.is_some();
    if !has_info {
        return model.id.clone();
    }

    let mut line = format!("{:<width$}", model.id, width = widths.name);
    if model.context.is_some() || model.max_output.is_some() {
        let ctx = model.context.as_deref().unwrap_or("?");
        let out = model.max_output.as_deref().unwrap_or("?");
        line.push_str(&format!(
            "  {}",
            style::dim(format!(
                "{:>cw$} ctx \u{00b7} {:>ow$} out",
                ctx,
                out,
                cw = widths.context.max(1),
                ow = widths.max_output.max(1),
            ))
        ));
    }
    if let (Some(input), Some(output)) = (&model.input_price, &model.output_price) {
        line.push_str(&format!(
            "  {}",
            style::dim(format!(
                "{:>iw$}/{}",
                input,
                output,
                iw = widths.input_price.max(1),
            ))
        ));
    }
    if let Some(mult) = model.multiplier {
        line.push_str(&format!("  {}", style::dim(format_multiplier(mult))));
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

/// Like `fetch_models`, but keeps image/audio models in the list. Used by
/// `aivo image`, where providers like xai advertise `grok-2-image` /
/// `grok-imagine-image` that `is_text_chat_model` would otherwise strip.
pub(crate) async fn fetch_all_models(client: &Client, key: &ApiKey) -> Result<Vec<String>> {
    fetch_models_detailed_filtered(client, key, false)
        .await
        .map(|v| v.into_iter().map(|m| m.id).collect())
}

/// Pulls per-model context-window metadata out of a detailed model list
/// so it can be persisted alongside the cached name list. Skips entries
/// where the provider didn't return a context window.
fn build_metadata_map(models: &[ModelInfo]) -> HashMap<String, ModelMetadata> {
    models
        .iter()
        .filter_map(|m| {
            m.context_tokens.map(|ctx| {
                (
                    m.id.clone(),
                    ModelMetadata {
                        context_window: Some(ctx),
                    },
                )
            })
        })
        .collect()
}

/// Cached variant of `fetch_all_models`.
pub(crate) async fn fetch_all_models_cached(
    client: &Client,
    key: &ApiKey,
    cache: &ModelsCache,
    bypass_cache: bool,
) -> Result<Vec<String>> {
    let is_ollama = crate::services::provider_profile::is_ollama_base(&key.base_url);
    let cache_key = full_catalog_key(&key.base_url);
    if !bypass_cache
        && !is_ollama
        && let Some(cached) = cache.get(&cache_key).await
    {
        return Ok(cached);
    }
    let models = fetch_all_models(client, key).await?;
    if !is_ollama {
        cache.set(&cache_key, models.clone()).await;
    }
    Ok(models)
}

/// Fetch models with full metadata from the API where available.
/// Providers like OpenRouter/Vercel return context window, pricing, and max output.
/// Google returns inputTokenLimit and outputTokenLimit.
/// Other providers return just IDs.
pub(crate) async fn fetch_models_detailed(client: &Client, key: &ApiKey) -> Result<Vec<ModelInfo>> {
    fetch_models_detailed_filtered(client, key, true).await
}

/// Implementation behind `fetch_models_detailed` / `fetch_all_models`. When
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
        ModelListingStrategy::Static(models) => Ok::<_, anyhow::Error>(
            models
                .iter()
                .map(|s| ModelInfo::id_only(s.to_string()))
                .collect(),
        ),
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
                anyhow::bail!("API returned {} — {}", status, body);
            }

            let resp: OpenAIModelsResponse = response.json().await?;
            Ok(resp.data.into_iter().map(|m| m.into_model_info()).collect())
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
                anyhow::bail!("Copilot API returned {} — {}", status, body);
            }

            let resp: OpenAIModelsResponse = response.json().await?;
            Ok(resp
                .data
                .into_iter()
                .filter(|m| is_copilot_chat_model(&m.id))
                .map(|m| m.into_model_info())
                .collect())
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
                anyhow::bail!("API returned {} — {}", status, body);
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
                        input_price: None,
                        output_price: None,
                        multiplier: None,
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
                anyhow::bail!("API returned {} — {}", status, body);
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
                    anyhow::bail!("API returned {} from {} — {}", status, url, body);
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
                    last_err = format!("API returned {} from {} — {}", status, url, body);
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
    if !bypass_cache
        && !is_ollama
        && let Some(cached) = cache.get(&key.base_url).await
    {
        return Ok(cached);
    }
    let models = fetch_models(client, key).await?;
    if !is_ollama {
        cache.set(&key.base_url, models.clone()).await;
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
        AIToolType::Codex => models
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
            routing_schema_version: 0,
            key: Zeroizing::new("sk-test".to_string()),
            created_at: "2026-01-01".to_string(),
        }
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
    fn test_tool_supports_default_empty_model_list() {
        assert!(!tool_supports_default_model(AIToolType::Claude, &[]));
        assert!(!tool_supports_default_model(AIToolType::Codex, &[]));
        assert!(!tool_supports_default_model(AIToolType::Gemini, &[]));
        assert!(!tool_supports_default_model(AIToolType::Pi, &[]));
        assert!(!tool_supports_default_model(AIToolType::Opencode, &[]));
    }
}
