//! Request patch pipeline for Aivo's Anthropic-compatible routing.
//!
//! This keeps provider-specific request quirks modular so routers stay focused on
//! transport and streaming.

use anyhow::Result;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Value, json};

use crate::services::effort;
use crate::services::model_names::transform_model_for_provider;

const ANTHROPIC_CACHE_CONTROL_BREAKPOINT_LIMIT: usize = 4;

pub struct RequestContext<'a> {
    pub upstream_base_url: &'a str,
    /// Provider's advertised model ids, when known, so `ModelNamePatch` can snap
    /// to an exact id. `None` falls back to the host-based transform.
    pub catalog: Option<&'a [String]>,
}

impl<'a> RequestContext<'a> {
    pub fn new(upstream_base_url: &'a str) -> Self {
        Self {
            upstream_base_url,
            catalog: None,
        }
    }

    pub fn with_catalog(mut self, catalog: Option<&'a [String]>) -> Self {
        self.catalog = catalog;
        self
    }
}

pub trait RequestPatch: Send + Sync {
    fn patch_json(&self, _route: &str, _body: &mut Value, _ctx: &RequestContext<'_>) -> Result<()> {
        Ok(())
    }

    fn patch_headers(
        &self,
        _route: &str,
        _headers: &mut HeaderMap,
        _ctx: &RequestContext<'_>,
    ) -> Result<()> {
        Ok(())
    }
}

pub struct RouterPipeline {
    patches: Vec<Box<dyn RequestPatch>>,
}

impl RouterPipeline {
    pub fn new(patches: Vec<Box<dyn RequestPatch>>) -> Self {
        Self { patches }
    }

    pub fn for_openrouter() -> Self {
        Self::new(vec![
            Box::new(CacheControlPatch),
            Box::new(ModelNamePatch),
            Box::new(AnthropicVersionPatch),
            Box::new(ThinkingNormalizationPatch),
        ])
    }

    pub fn patch_json(
        &self,
        route: &str,
        body: &mut Value,
        ctx: &RequestContext<'_>,
    ) -> Result<()> {
        for patch in &self.patches {
            patch.patch_json(route, body, ctx)?;
        }
        Ok(())
    }

    pub fn patch_headers(
        &self,
        route: &str,
        headers: &mut HeaderMap,
        ctx: &RequestContext<'_>,
    ) -> Result<()> {
        for patch in &self.patches {
            patch.patch_headers(route, headers, ctx)?;
        }
        Ok(())
    }
}

/// Normalizes provider model names (e.g. OpenRouter model prefix/version shape).
pub struct ModelNamePatch;

impl RequestPatch for ModelNamePatch {
    fn patch_json(&self, _route: &str, body: &mut Value, ctx: &RequestContext<'_>) -> Result<()> {
        if let Some(model) = body.get_mut("model")
            && let Some(model_str) = model.as_str()
        {
            *model = Value::String(transform_model_for_provider(
                ctx.catalog,
                ctx.upstream_base_url,
                model_str,
            ));
        }
        Ok(())
    }
}

/// Injects `cache_control` on the system prompt and last user message for Anthropic prompt caching.
pub struct CacheControlPatch;

impl RequestPatch for CacheControlPatch {
    fn patch_json(&self, route: &str, body: &mut Value, _ctx: &RequestContext<'_>) -> Result<()> {
        match route {
            "messages" => {
                if cache_control_breakpoint_count(body) < ANTHROPIC_CACHE_CONTROL_BREAKPOINT_LIMIT
                    && let Some(system) = body.get_mut("system")
                {
                    inject_cache_control_on_last_block(system);
                }
            }
            "chat/completions" => {
                inject_chat_completions_cache_control(body);
                return Ok(());
            }
            _ => return Ok(()),
        }

        if cache_control_breakpoint_count(body) < ANTHROPIC_CACHE_CONTROL_BREAKPOINT_LIMIT
            && let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut())
        {
            for msg in messages.iter_mut().rev() {
                if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
                    continue;
                }
                if let Some(content) = msg.get_mut("content") {
                    inject_cache_control_on_last_block(content);
                }
                break;
            }
        }
        Ok(())
    }
}

/// Inject `cache_control` markers on an OpenAI Chat Completions request body.
/// Adds markers to the system message and last user message.
pub(crate) fn inject_chat_completions_cache_control(body: &mut Value) {
    if cache_control_breakpoint_count(body) < ANTHROPIC_CACHE_CONTROL_BREAKPOINT_LIMIT
        && let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut())
    {
        for msg in messages.iter_mut().rev() {
            if msg.get("role").and_then(|r| r.as_str()) == Some("system") {
                if let Some(content) = msg.get_mut("content") {
                    inject_cache_control_on_last_block(content);
                }
                break;
            }
        }
    }
    if cache_control_breakpoint_count(body) < ANTHROPIC_CACHE_CONTROL_BREAKPOINT_LIMIT
        && let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut())
    {
        for msg in messages.iter_mut().rev() {
            if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
                continue;
            }
            if let Some(content) = msg.get_mut("content") {
                inject_cache_control_on_last_block(content);
            }
            break;
        }
    }
}

fn cache_control_breakpoint_count(value: &Value) -> usize {
    match value {
        Value::Object(obj) => {
            let here = obj
                .get("cache_control")
                .and_then(Value::as_object)
                .and_then(|cc| cc.get("type"))
                .and_then(Value::as_str)
                .is_some_and(|kind| kind == "ephemeral") as usize;
            here + obj
                .values()
                .map(cache_control_breakpoint_count)
                .sum::<usize>()
        }
        Value::Array(arr) => arr.iter().map(cache_control_breakpoint_count).sum(),
        _ => 0,
    }
}

/// Recursively remove `cache_control` keys from any object inside `body`.
/// Used for upstreams that reject Anthropic-specific cache_control on
/// system/message content (e.g., Bedrock-style shims). Walks the JSON tree
/// rather than enumerating known sites so future schema additions stay safe.
pub(crate) fn strip_cache_control(body: &mut Value) {
    match body {
        Value::Object(obj) => {
            obj.remove("cache_control");
            for (_k, v) in obj.iter_mut() {
                strip_cache_control(v);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                strip_cache_control(v);
            }
        }
        _ => {}
    }
}

pub(crate) fn inject_cache_control_on_last_block(value: &mut Value) {
    match value {
        Value::String(s) => {
            let text = s.clone();
            *value = json!([{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"}
            }]);
        }
        Value::Array(blocks) => {
            if let Some(last) = blocks.last_mut()
                && last.get("cache_control").is_none()
            {
                last["cache_control"] = json!({"type": "ephemeral"});
            }
        }
        _ => {}
    }
}

/// Adds Anthropic API version header where required by Anthropic-format endpoints.
pub struct AnthropicVersionPatch;

impl RequestPatch for AnthropicVersionPatch {
    fn patch_headers(
        &self,
        route: &str,
        headers: &mut HeaderMap,
        _ctx: &RequestContext<'_>,
    ) -> Result<()> {
        if matches!(route, "messages" | "messages/count_tokens") {
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        }
        Ok(())
    }
}

/// Reconciles `thinking` with per-model capabilities (version cuts delegate to
/// `crate::services::effort`, so this native path and the bridge stay in sync):
/// strip adaptive where unsupported (with its paired effort), rewrite
/// `enabled`+`budget_tokens` → `adaptive`, and omit `disabled` on Fable/Mythos.
pub struct ThinkingNormalizationPatch;

impl RequestPatch for ThinkingNormalizationPatch {
    fn patch_json(&self, _route: &str, body: &mut Value, _ctx: &RequestContext<'_>) -> Result<()> {
        let Some(obj) = body.as_object_mut() else {
            return Ok(());
        };
        let Some(model) = obj
            .get("model")
            .and_then(|m| m.as_str())
            .map(str::to_string)
        else {
            return Ok(());
        };
        let Some(type_str) = obj
            .get("thinking")
            .and_then(|t| t.get("type"))
            .and_then(|t| t.as_str())
            .map(str::to_string)
        else {
            return Ok(());
        };

        match type_str.as_str() {
            "adaptive" if !effort::anthropic_supports_adaptive_thinking(&model) => {
                obj.remove("thinking");
                if !effort::anthropic_supports_output_effort(&model) {
                    drop_output_config_effort(obj);
                }
            }
            "enabled" if effort::anthropic_thinking_uses_adaptive(&model) => {
                rewrite_enabled_to_adaptive(obj);
            }
            "disabled" if effort::anthropic_rejects_disabled_thinking(&model) => {
                obj.remove("thinking");
            }
            _ => {}
        }

        Ok(())
    }
}

fn drop_output_config_effort(obj: &mut serde_json::Map<String, Value>) {
    let drop_output_config = obj
        .get_mut("output_config")
        .and_then(|o| o.as_object_mut())
        .map(|oc| {
            oc.remove("effort");
            oc.is_empty()
        })
        .unwrap_or(false);
    if drop_output_config {
        obj.remove("output_config");
    }
}

/// Preserves non-`type`/non-`budget_tokens` keys such as `display`, and
/// drops `budget_tokens` (adaptive carries no budget). Does not synthesize
/// `output_config.effort` from the dropped budget — many upstreams reject
/// `output_config` outright, so introducing the field can produce "Extra
/// inputs are not permitted" 400s. Callers that want effort control on
/// Opus 4.7 can set `output_config.effort` themselves.
fn rewrite_enabled_to_adaptive(obj: &mut serde_json::Map<String, Value>) {
    let mut new_thinking = match obj.remove("thinking") {
        Some(Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    new_thinking.remove("budget_tokens");
    new_thinking.insert("type".to_string(), Value::String("adaptive".to_string()));
    obj.insert("thinking".to_string(), Value::Object(new_thinking));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_name_patch_openrouter_transform() {
        let patch = ModelNamePatch;
        let mut body = serde_json::json!({"model":"claude-sonnet-4-6"});
        let ctx = RequestContext::new("https://openrouter.ai/api/v1");
        patch.patch_json("messages", &mut body, &ctx).unwrap();
        assert_eq!(body["model"], "anthropic/claude-sonnet-4.6");
    }

    #[test]
    fn test_model_name_patch_non_openrouter_passthrough() {
        let patch = ModelNamePatch;
        let mut body = serde_json::json!({"model":"claude-sonnet-4-6"});
        let ctx = RequestContext::new("https://api.example.com/v1");
        patch.patch_json("messages", &mut body, &ctx).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-6");
    }

    #[test]
    fn test_anthropic_version_patch_only_messages_routes() {
        let patch = AnthropicVersionPatch;
        let ctx = RequestContext::new("https://openrouter.ai/api/v1");

        let mut headers = HeaderMap::new();
        patch.patch_headers("messages", &mut headers, &ctx).unwrap();
        assert!(headers.get("anthropic-version").is_some());

        let mut headers = HeaderMap::new();
        patch
            .patch_headers("chat/completions", &mut headers, &ctx)
            .unwrap();
        assert!(headers.get("anthropic-version").is_none());
    }

    #[test]
    fn test_cache_control_patch_converts_string_system_to_block() {
        let patch = CacheControlPatch;
        let mut body = serde_json::json!({
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let ctx = RequestContext::new("https://api.anthropic.com/v1");
        patch.patch_json("messages", &mut body, &ctx).unwrap();

        let system = body["system"].as_array().unwrap();
        assert_eq!(system.len(), 1);
        assert_eq!(system[0]["type"], "text");
        assert_eq!(system[0]["text"], "You are helpful.");
        assert_eq!(system[0]["cache_control"]["type"], "ephemeral");

        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_cache_control_patch_adds_to_existing_blocks() {
        let patch = CacheControlPatch;
        let mut body = serde_json::json!({
            "system": [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
                {"role": "user", "content": [{"type": "text", "text": "Bye"}]}
            ]
        });
        let ctx = RequestContext::new("https://api.anthropic.com/v1");
        patch.patch_json("messages", &mut body, &ctx).unwrap();

        // Only last system block gets cache_control
        assert!(body["system"][0].get("cache_control").is_none());
        assert_eq!(body["system"][1]["cache_control"]["type"], "ephemeral");

        // Only last user message gets cache_control
        assert!(
            body["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            body["messages"][2]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn test_cache_control_patch_preserves_existing_cache_control() {
        let patch = CacheControlPatch;
        let mut body = serde_json::json!({
            "system": [{"type": "text", "text": "Sys", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}]}]
        });
        let ctx = RequestContext::new("https://api.anthropic.com/v1");
        patch.patch_json("messages", &mut body, &ctx).unwrap();

        // Should not double-add
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn cache_control_patch_does_not_add_fifth_breakpoint_to_hoisted_system() {
        let patch = CacheControlPatch;
        let mut body = serde_json::json!({
            "system": [
                {"type": "text", "text": "Base A", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Base B", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Base C", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "system", "content": [
                    {"type": "text", "text": "Hoisted catalog", "cache_control": {"type": "ephemeral"}}
                ]}
            ]
        });
        let ctx = RequestContext::new("https://api.anthropic.com/v1");

        let report =
            crate::services::anthropic_chat_request::hoist_anthropic_system_messages(&mut body)
                .expect("system message hoisted");
        assert_eq!(report.hoisted_blocks, 1);
        assert_eq!(cache_control_breakpoint_count(&body), 4);

        patch.patch_json("messages", &mut body, &ctx).unwrap();

        let system = body["system"].as_array().unwrap();
        assert_eq!(cache_control_breakpoint_count(&body), 4);
        assert!(
            system
                .last()
                .expect("hoisted block appended")
                .get("cache_control")
                .is_none(),
            "hoisted block must not receive a fifth cache_control marker"
        );
    }

    #[test]
    fn test_cache_control_patch_chat_completions_system_message() {
        let patch = CacheControlPatch;
        let mut body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"}
            ]
        });
        let ctx = RequestContext::new("https://api.anthropic.com/v1");
        patch
            .patch_json("chat/completions", &mut body, &ctx)
            .unwrap();

        // System message content converted to block with cache_control
        let sys_content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(sys_content[0]["cache_control"]["type"], "ephemeral");

        // Last user message also gets cache_control
        let user_content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(user_content[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn chat_completions_cache_control_respects_breakpoint_limit() {
        let mut body = serde_json::json!({
            "messages": [
                {"role": "system", "content": [
                    {"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "B", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "C", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "D", "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": "Hi"}
            ]
        });

        inject_chat_completions_cache_control(&mut body);

        assert_eq!(cache_control_breakpoint_count(&body), 4);
        assert!(
            body["messages"][1]["content"].is_string(),
            "user message should stay unmodified once the request is at the limit"
        );
    }

    #[test]
    fn test_cache_control_patch_skips_unknown_routes() {
        let patch = CacheControlPatch;
        let mut body = serde_json::json!({"system": "Hello", "messages": []});
        let ctx = RequestContext::new("https://api.anthropic.com/v1");
        patch
            .patch_json("messages/count_tokens", &mut body, &ctx)
            .unwrap();
        assert!(body["system"].is_string());
    }

    #[test]
    fn test_cache_control_chat_completions_multiple_system_messages() {
        let patch = CacheControlPatch;
        let mut body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "First system."},
                {"role": "system", "content": "Second system."},
                {"role": "user", "content": "Hi"}
            ]
        });
        let ctx = RequestContext::new("https://api.anthropic.com/v1");
        patch
            .patch_json("chat/completions", &mut body, &ctx)
            .unwrap();

        // First system should NOT have cache_control
        assert!(
            body["messages"][0]["content"].is_string(),
            "first system message should remain a plain string"
        );
        // Last system message SHOULD have cache_control
        let last_sys = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(last_sys[0]["cache_control"]["type"], "ephemeral");
        // User message should also have cache_control
        let user = body["messages"][2]["content"].as_array().unwrap();
        assert_eq!(user[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn strip_cache_control_recursively_removes_nested_keys() {
        let mut body = serde_json::json!({
            "messages": [
                {"role": "system", "content": [
                    {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "trailing"}
                ]}
            ],
            "system": [
                {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
            ]
        });
        super::strip_cache_control(&mut body);
        // Top-level system block stripped.
        assert!(body["system"][0].get("cache_control").is_none());
        // Nested message-level blocks stripped.
        assert!(
            body["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body["messages"][1]["content"][0]
                .get("cache_control")
                .is_none()
        );
        // Other fields untouched.
        assert_eq!(body["messages"][0]["content"][0]["text"], "sys");
        assert_eq!(body["messages"][1]["content"][1]["text"], "trailing");
    }

    #[test]
    fn test_pipeline_applies_all_patches() {
        let pipeline = RouterPipeline::for_openrouter();
        let ctx = RequestContext::new("https://openrouter.ai/api/v1");
        let mut body = serde_json::json!({"model":"claude-haiku-4-5"});
        let mut headers = HeaderMap::new();

        pipeline.patch_json("messages", &mut body, &ctx).unwrap();
        pipeline
            .patch_headers("messages", &mut headers, &ctx)
            .unwrap();

        assert_eq!(body["model"], "anthropic/claude-haiku-4.5");
        assert_eq!(
            headers
                .get("anthropic-version")
                .and_then(|v| v.to_str().ok()),
            Some("2023-06-01")
        );
    }

    // ─── ThinkingNormalizationPatch ───────────────────────────────────────

    fn run_thinking_patch(body: &mut Value) {
        let ctx = RequestContext::new("https://openrouter.ai/api/v1");
        ThinkingNormalizationPatch
            .patch_json("messages", body, &ctx)
            .unwrap();
    }

    #[test]
    fn thinking_patch_keeps_adaptive_on_opus_4_7() {
        let mut body = json!({
            "model": "claude-opus-4-7",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }

    #[test]
    fn thinking_patch_keeps_adaptive_on_anthropic_prefixed_sonnet_4_6() {
        // Mirrors what arrives after ModelNamePatch rewrites for OpenRouter:
        // dotted form, anthropic/ prefix.
        let mut body = json!({
            "model": "anthropic/claude-sonnet-4.6",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }

    #[test]
    fn thinking_patch_keeps_adaptive_on_opus_4_6_with_date_suffix() {
        let mut body = json!({
            "model": "claude-opus-4-6-20260120",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }

    #[test]
    fn thinking_patch_keeps_adaptive_on_mythos() {
        let mut body = json!({
            "model": "claude-mythos-preview",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }

    #[test]
    fn thinking_patch_tolerates_dotted_canonical_form() {
        let mut body = json!({
            "model": "claude-sonnet-4.6",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }

    #[test]
    fn thinking_patch_keeps_adaptive_on_bedrock_style_sonnet_4_6() {
        let mut body = json!({
            "model": "us.anthropic.claude-sonnet-4-6",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }

    #[test]
    fn thinking_patch_strips_adaptive_on_haiku_4_5() {
        let mut body = json!({
            "model": "claude-haiku-4-5-20251001",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn thinking_patch_strips_adaptive_on_anthropic_prefixed_haiku() {
        let mut body = json!({
            "model": "anthropic/claude-haiku-4-5-20251001",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn thinking_patch_strips_adaptive_on_sonnet_4_5() {
        let mut body = json!({
            "model": "claude-sonnet-4-5",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn thinking_patch_preserves_effort_on_opus_4_5_when_stripping_adaptive() {
        let mut body = json!({
            "model": "claude-opus-4-5",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"}
        });
        run_thinking_patch(&mut body);
        assert!(body.get("thinking").is_none());
        assert_eq!(body["output_config"], json!({"effort": "low"}));
    }

    #[test]
    fn thinking_patch_strips_paired_output_config_effort() {
        let mut body = json!({
            "model": "claude-haiku-4-5",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"}
        });
        run_thinking_patch(&mut body);
        assert!(body.get("thinking").is_none());
        // output_config is removed entirely once it becomes empty — many
        // upstreams reject `output_config` on non-adaptive-capable models.
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn thinking_patch_preserves_other_output_config_keys_when_stripping_adaptive() {
        let mut body = json!({
            "model": "claude-haiku-4-5",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high", "other_key": "keep"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["output_config"], json!({"other_key": "keep"}));
    }

    #[test]
    fn thinking_patch_rewrites_enabled_to_adaptive_on_opus_4_7() {
        let mut body = json!({
            "model": "claude-opus-4-7",
            "thinking": {"type": "enabled", "budget_tokens": 16000}
        });
        run_thinking_patch(&mut body);
        // budget_tokens dropped (adaptive carries none); no output_config
        // synthesized — the field is rejected as "Extra inputs" on many
        // upstreams.
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn thinking_patch_preserves_display_when_rewriting_enabled_on_opus_4_7() {
        let mut body = json!({
            "model": "claude-opus-4-7",
            "thinking": {
                "type": "enabled",
                "budget_tokens": 4096,
                "display": "summarized"
            }
        });
        run_thinking_patch(&mut body);
        assert_eq!(
            body["thinking"],
            json!({"type": "adaptive", "display": "summarized"})
        );
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn thinking_patch_leaves_user_output_config_alone_when_rewriting_enabled() {
        // We don't touch output_config in the rewrite — if the user set
        // effort explicitly, it survives.
        let mut body = json!({
            "model": "claude-opus-4-7",
            "thinking": {"type": "enabled", "budget_tokens": 16000},
            "output_config": {"effort": "low"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
        assert_eq!(body["output_config"], json!({"effort": "low"}));
    }

    #[test]
    fn thinking_patch_keeps_enabled_on_sonnet_4_6() {
        let mut body = json!({
            "model": "claude-sonnet-4-6",
            "thinking": {"type": "enabled", "budget_tokens": 8192}
        });
        run_thinking_patch(&mut body);
        assert_eq!(
            body["thinking"],
            json!({"type": "enabled", "budget_tokens": 8192})
        );
    }

    #[test]
    fn thinking_patch_keeps_enabled_on_haiku_4_5() {
        let mut body = json!({
            "model": "claude-haiku-4-5",
            "thinking": {"type": "enabled", "budget_tokens": 4096}
        });
        run_thinking_patch(&mut body);
        assert_eq!(
            body["thinking"],
            json!({"type": "enabled", "budget_tokens": 4096})
        );
    }

    #[test]
    fn thinking_patch_ignores_disabled_type() {
        let mut body = json!({
            "model": "claude-haiku-4-5",
            "thinking": {"type": "disabled"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "disabled"}));
    }

    #[test]
    fn thinking_patch_no_op_without_thinking_field() {
        let mut body = json!({"model": "claude-haiku-4-5", "messages": []});
        run_thinking_patch(&mut body);
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn thinking_patch_no_op_without_model_field() {
        let mut body = json!({"thinking": {"type": "adaptive"}});
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }

    #[test]
    fn thinking_patch_keeps_adaptive_on_opus_4_8() {
        let mut body = json!({
            "model": "claude-opus-4-8",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }

    #[test]
    fn thinking_patch_keeps_adaptive_on_fable() {
        let mut body = json!({
            "model": "claude-fable-5",
            "thinking": {"type": "adaptive"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }

    #[test]
    fn thinking_patch_rewrites_enabled_to_adaptive_on_opus_4_8_and_fable() {
        for model in ["claude-opus-4-8", "claude-fable-5"] {
            let mut body = json!({
                "model": model,
                "thinking": {"type": "enabled", "budget_tokens": 16000}
            });
            run_thinking_patch(&mut body);
            assert_eq!(body["thinking"], json!({"type": "adaptive"}), "{model}");
        }
    }

    #[test]
    fn thinking_patch_omits_disabled_on_fable() {
        let mut body = json!({
            "model": "claude-fable-5",
            "thinking": {"type": "disabled"}
        });
        run_thinking_patch(&mut body);
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn thinking_patch_keeps_disabled_on_opus_4_8() {
        let mut body = json!({
            "model": "claude-opus-4-8",
            "thinking": {"type": "disabled"}
        });
        run_thinking_patch(&mut body);
        assert_eq!(body["thinking"], json!({"type": "disabled"}));
    }

    #[test]
    fn full_pipeline_preserves_adaptive_on_opus_4_7_through_openrouter_rename() {
        // End-to-end: Claude Code sends dash-form, OpenRouter pipeline
        // renames the model to anthropic-prefixed dotted form, and the
        // thinking patch (running after the rename) must still recognize
        // Opus 4.7 as supporting adaptive.
        let pipeline = RouterPipeline::for_openrouter();
        let ctx = RequestContext::new("https://openrouter.ai/api/v1");
        let mut body = json!({
            "model": "claude-opus-4-7",
            "thinking": {"type": "adaptive"}
        });
        pipeline.patch_json("messages", &mut body, &ctx).unwrap();

        assert_eq!(body["model"], "anthropic/claude-opus-4.7");
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }

    #[test]
    fn full_pipeline_strips_adaptive_on_haiku_4_5_through_openrouter_rename() {
        let pipeline = RouterPipeline::for_openrouter();
        let ctx = RequestContext::new("https://openrouter.ai/api/v1");
        let mut body = json!({
            "model": "claude-haiku-4-5",
            "thinking": {"type": "adaptive"}
        });
        pipeline.patch_json("messages", &mut body, &ctx).unwrap();

        assert_eq!(body["model"], "anthropic/claude-haiku-4.5");
        assert!(body.get("thinking").is_none());
    }
}
