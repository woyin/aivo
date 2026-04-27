//! Centralized default values used across the protocol-bridge layer.
//!
//! Each entry was previously hardcoded inline at one or more bridge sites.
//! Putting them here makes the rationale auditable in one place and avoids
//! the drift that crept in (e.g. multiple `"chatcmpl-aivo"` literals at
//! 4 different file:line locations).

/// Synthetic OpenAI chat-completion ID used when the upstream response
/// (Anthropic, Gemini, Copilot) omits its own ID. OpenAI clients require
/// `id` to be present and non-empty on every chunk; this string lets them
/// tell the response is from aivo's bridge rather than a real OpenAI ID.
pub(crate) const BRIDGE_FALLBACK_OPENAI_RESPONSE_ID: &str = "chatcmpl-aivo";

/// Default `max_tokens` to emit on Anthropic requests when the OpenAI-shaped
/// caller didn't set one. Anthropic's `/v1/messages` rejects requests
/// without `max_tokens`, so the bridge has to substitute a value.
///
/// Raised from 4096 → 16384 because Claude 4 supports up to 64k output and
/// the older 4096 silently truncated long responses through the bridge.
pub(crate) const BRIDGE_DEFAULT_ANTHROPIC_MAX_TOKENS: u64 = 16_384;
