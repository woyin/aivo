//! Library exports for the aivo CLI.
//! Re-exports all public modules for testing and library use.

pub mod agent;
pub mod cli;
pub mod cli_args;
pub mod commands;
pub mod constants;
pub mod errors;
pub mod key_resolution;
pub mod plugin;
pub mod run;
pub mod services;
pub mod style;
pub mod tui;
pub mod version;

pub use errors::{CLIError, ErrorCategory, ExitCode};

// Same sandbox the integration binaries use (see that file for rationale).
#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod test_sandbox;
