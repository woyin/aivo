//! Version management for the aivo CLI.
//! Version is embedded at build time from Cargo.toml.

/// The version of the aivo CLI, embedded at build time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
