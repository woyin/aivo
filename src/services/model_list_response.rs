use serde_json::{Value, json};

use super::model_metadata::ResolvedLimits;

/// One row of a `/models` response. `limits` comes from the
/// `model_metadata::resolve_limits` cascade; all-`None` (the default) emits
/// no limit fields.
pub(crate) struct ModelListEntry {
    pub id: String,
    pub owned_by: String,
    pub limits: ResolvedLimits,
}

/// Builds a `/models` response body that satisfies both OpenAI-compatible
/// clients and codex 0.132+'s stricter models manager.
///
/// OpenAI-compatible clients read `{"object":"list","data":[...]}`. Codex's
/// `ModelsResponse` reads `{"models":[<ModelInfo>...]}` and ignores unknown
/// fields, so both shapes can live in one response. Known limits are emitted
/// as `context_length`/`max_output_tokens` (OpenRouter's field names, which
/// aivo's own `/v1/models` harvester also picks up) in the OpenAI shape and
/// as `context_window`/`max_context_window`/`max_output_tokens` in codex's.
pub(crate) fn build_models_response_body(entries: Vec<ModelListEntry>) -> Value {
    let openai_entries: Vec<Value> = entries
        .iter()
        .map(|entry| {
            let mut row = json!({
                "id": entry.id,
                "object": "model",
                "owned_by": entry.owned_by,
            });
            if let Some(context) = entry.limits.context {
                row["context_length"] = json!(context);
            }
            if let Some(output) = entry.limits.output {
                row["max_output_tokens"] = json!(output);
            }
            row
        })
        .collect();
    let codex_entries: Vec<Value> = entries
        .iter()
        .map(|entry| codex_model_info(&entry.id, &entry.limits))
        .collect();
    json!({
        "object": "list",
        "data": openai_entries,
        "models": codex_entries,
    })
}

pub(crate) fn build_models_response_body_for_owner(models: &[String], owned_by: &str) -> Value {
    build_models_response_body(
        models
            .iter()
            .map(|id| ModelListEntry {
                id: id.clone(),
                owned_by: owned_by.to_string(),
                limits: ResolvedLimits::default(),
            })
            .collect(),
    )
}

/// Minimal valid `ModelInfo` for codex. Every field serde considers required
/// (no `#[serde(default)]`, no `Option` with default) is present; optional
/// fields are omitted so codex's defaults apply.
fn codex_model_info(id: &str, limits: &ResolvedLimits) -> Value {
    let mut info = json!({
        "slug": id,
        "display_name": id,
        "description": null,
        "supported_reasoning_levels": [],
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 0,
        "availability_nux": null,
        "upgrade": null,
        "base_instructions": "",
        "supports_reasoning_summaries": false,
        "support_verbosity": false,
        "default_verbosity": null,
        "apply_patch_tool_type": null,
        "truncation_policy": {"mode": "tokens", "limit": 100_000},
        "supports_parallel_tool_calls": false,
        "experimental_supported_tools": [],
    });
    if let Some(context) = limits.context {
        info["context_window"] = json!(context);
        info["max_context_window"] = json!(context);
    }
    if let Some(output) = limits.output {
        info["max_output_tokens"] = json!(output);
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_response_body_satisfies_openai_and_codex_consumers() {
        let body = build_models_response_body(vec![
            ModelListEntry {
                id: "composer-2.5".to_string(),
                owned_by: "cursor".to_string(),
                limits: ResolvedLimits::default(),
            },
            ModelListEntry {
                id: "auto".to_string(),
                owned_by: "cursor".to_string(),
                limits: ResolvedLimits::default(),
            },
        ]);

        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().expect("data must be an array");
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "composer-2.5");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["owned_by"], "cursor");

        let models = body["models"].as_array().expect("models must be an array");
        assert_eq!(models.len(), 2);
        let first = &models[0];
        for required in [
            "slug",
            "display_name",
            "description",
            "supported_reasoning_levels",
            "shell_type",
            "visibility",
            "supported_in_api",
            "priority",
            "availability_nux",
            "upgrade",
            "base_instructions",
            "supports_reasoning_summaries",
            "support_verbosity",
            "default_verbosity",
            "apply_patch_tool_type",
            "truncation_policy",
            "supports_parallel_tool_calls",
            "experimental_supported_tools",
        ] {
            assert!(
                first.get(required).is_some(),
                "codex requires field `{required}`: {first:?}"
            );
        }
        assert_eq!(first["slug"], "composer-2.5");
        assert_eq!(first["truncation_policy"]["mode"], "tokens");
    }

    #[test]
    fn known_limits_enrich_both_shapes() {
        let body = build_models_response_body(vec![
            ModelListEntry {
                id: "claude-sonnet-4-6".to_string(),
                owned_by: "aivo".to_string(),
                limits: ResolvedLimits {
                    context: Some(1_000_000),
                    output: Some(64_000),
                    caps: None,
                    image_input: None,
                    reasoning_efforts: Vec::new(),
                },
            },
            ModelListEntry {
                id: "mystery-model".to_string(),
                owned_by: "aivo".to_string(),
                limits: ResolvedLimits::default(),
            },
        ]);

        let enriched = &body["data"][0];
        assert_eq!(enriched["context_length"], 1_000_000);
        assert_eq!(enriched["max_output_tokens"], 64_000);
        let codex = &body["models"][0];
        assert_eq!(codex["context_window"], 1_000_000);
        assert_eq!(codex["max_context_window"], 1_000_000);
        assert_eq!(codex["max_output_tokens"], 64_000);

        // Unknown models omit every limit field rather than emitting nulls.
        let bare = &body["data"][1];
        assert!(bare.get("context_length").is_none());
        assert!(bare.get("max_output_tokens").is_none());
        let bare_codex = &body["models"][1];
        assert!(bare_codex.get("context_window").is_none());
        assert!(bare_codex.get("max_context_window").is_none());
        assert!(bare_codex.get("max_output_tokens").is_none());
    }
}
