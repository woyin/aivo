//! aivo's native agent, driven by `aivo code`. The tool-using loop runs
//! in-process in `engine`: it composes OpenAI chat requests, calls the model
//! through the loopback serve (the sole network egress), executes `tools`
//! locally, and renders through an `AgentUi` (the chat TUI). `protocol` holds
//! the shared data types; `serve_client` is the streaming provider call.

pub mod apply_patch;
pub mod ask;
pub mod checkpoint;
pub mod compaction;
pub mod engine;
pub mod file_tracker;
pub mod grant_store;
pub mod guards;
pub mod hooks;
pub mod jobs;
pub mod lsp;
pub mod mcp;
pub mod mcp_import;
pub mod notes;
pub mod packs;
pub mod plan;
pub mod plan_mode;
pub mod protocol;
pub mod request;
pub mod retry;
pub mod review;
pub mod sandbox;
pub mod secrets_guard;
pub mod serve_client;
pub mod skills;
pub mod subagents;
pub mod system_prompt;
pub mod tokens;
pub mod tool_repair;
pub mod tools;
pub mod verify;
