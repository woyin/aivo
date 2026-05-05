//! Library exports for the aivo CLI.
//! Re-exports all public modules for testing and library use.

pub mod cli;
pub mod cli_args;
pub mod commands;
pub mod constants;
pub mod errors;
pub mod key_resolution;
pub mod run;
pub mod services;
pub mod style;
pub mod tui;
pub mod version;

pub use errors::{CLIError, ErrorCategory, ExitCode};
