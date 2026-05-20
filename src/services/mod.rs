//! Core services for the aivo CLI. The folder is flat; tiers below.
//!
//! - Spine: session/key storage, crypto, http, system_env.
//! - Launch: ai_launcher + env injection + shadow homes for OAuth tools.
//! - Routers / bridges: local HTTP proxies and protocol shape translation.
//! - Provider metadata: known_providers, provider_protocol, provider_profile,
//!   model_names, model_compat, models_cache. Check these before adding a
//!   new `base_url.contains(...)` branch.
//! - Provider glue, optional features (audio/image/video/share), utilities.

pub mod ai_launcher;
pub mod amp_bridge;
pub mod amp_threads;
pub mod amp_trust;
pub mod anthropic_chat_request;
pub mod anthropic_chat_response;
pub mod anthropic_route_pipeline;
pub mod anthropic_router;
pub mod anthropic_to_openai_router;
pub mod api_key_store;
pub mod atomic_write;
pub mod audio_cache;
pub mod audio_gen;
pub mod bridge_defaults;
pub mod browser_open;
pub mod chat_session_store;
pub mod claude_oauth;
pub mod codex_home_shadow;
pub mod codex_model_map;
pub mod codex_oauth;
pub mod codex_oauth_callback;
pub mod context_ingest;
pub mod context_render;
pub mod context_window;
pub mod copilot_auth;
pub mod copilot_router;
pub mod device_fingerprint;
#[cfg(target_env = "musl")]
pub mod dns_resolver;
pub mod effort;
pub mod environment_injector;
pub mod export_crypto;
pub mod gemini_home_shadow;
pub mod gemini_oauth;
pub mod gemini_oauth_callback;
pub mod gemini_router;
pub mod global_stats;
pub mod http_debug;
pub mod http_utils;
pub mod huggingface;
pub mod id_compact;
pub mod image_gen;
pub mod key_compat;
pub mod known_providers;
pub mod last_selection;
pub mod launch_args;
pub mod launch_runtime;
pub mod log_store;
pub mod media_io;
pub mod model_compat;
pub mod model_names;
pub mod models_cache;
pub mod native_session_probe;
pub mod oauth_relogin;
pub mod ollama;
pub mod openai_anthropic_bridge;
pub mod openai_gemini_bridge;
pub mod openai_models;
pub mod path_search;
pub mod percent_codec;
pub mod playback;
pub mod project_id;
pub mod protocol_fallback;
pub mod provider_profile;
pub mod provider_protocol;
pub mod request_log;
pub mod responses_chat_conversion;
pub mod responses_to_chat_router;
pub mod serve_responses;
pub mod serve_router;
pub mod serve_stream_converters;
pub mod serve_upstream;
pub mod session_crypto;
pub mod session_store;
pub mod share_codec;
pub mod share_local_server;
pub mod share_payload;
pub mod share_picker;
pub mod share_redact;
pub mod share_resolver;
pub mod share_tunnel;
pub mod shutdown_signal;
pub mod since;
pub mod stdin_io;
pub mod symlink_util;
pub mod system_env;
pub mod terminal_graphics;
pub mod termux_exec;
pub mod usage_stats_store;
pub mod video_gen;

pub use ai_launcher::AILauncher;
pub use anthropic_router::{AnthropicRouter, AnthropicRouterConfig};
pub use anthropic_to_openai_router::{AnthropicToOpenAIRouter, AnthropicToOpenAIRouterConfig};
pub use copilot_router::{CopilotRouter, CopilotRouterConfig};
pub use environment_injector::EnvironmentInjector;
pub use gemini_router::{GeminiRouter, GeminiRouterConfig};
pub use models_cache::ModelsCache;
pub use responses_to_chat_router::{ResponsesToChatRouter, ResponsesToChatRouterConfig};
pub use session_store::SessionStore;
