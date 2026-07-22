//! Centralized error handling module for the aivo CLI.
//! Defines error types, exit codes, and error classification utilities.
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitCode {
    Success,
    UserError,
    NetworkError,
    AuthError,
    ToolExit(i32),
}

impl ExitCode {
    pub fn code(self) -> i32 {
        match self {
            ExitCode::Success => 0,
            ExitCode::UserError => 1,
            ExitCode::NetworkError => 2,
            ExitCode::AuthError => 3,
            ExitCode::ToolExit(n) => n,
        }
    }

    /// Severity rank (higher = worse) for [`worse_of`](Self::worse_of).
    fn severity(self) -> u8 {
        match self {
            ExitCode::Success => 0,
            ExitCode::UserError => 1,
            ExitCode::NetworkError => 2,
            ExitCode::AuthError => 3,
            ExitCode::ToolExit(_) => 4,
        }
    }

    /// The worse (higher-severity) of two codes.
    pub fn worse_of(self, other: ExitCode) -> ExitCode {
        if other.severity() > self.severity() {
            other
        } else {
            self
        }
    }
}

impl From<ExitCode> for i32 {
    fn from(code: ExitCode) -> Self {
        code.code()
    }
}

impl fmt::Display for ExitCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    User,
    Network,
    Auth,
}

impl ErrorCategory {
    pub fn exit_code(self) -> ExitCode {
        match self {
            ErrorCategory::User => ExitCode::UserError,
            ErrorCategory::Network => ExitCode::NetworkError,
            ErrorCategory::Auth => ExitCode::AuthError,
        }
    }
}

/// CLI error with category for exit code mapping.
#[derive(Debug)]
pub struct CLIError {
    message: String,
    category: ErrorCategory,
    details: Option<String>,
    suggestion: Option<String>,
}

impl CLIError {
    pub fn new(
        message: impl Into<String>,
        category: ErrorCategory,
        details: Option<impl Into<String>>,
        suggestion: Option<impl Into<String>>,
    ) -> Self {
        Self {
            message: message.into(),
            category,
            details: details.map(|d| d.into()),
            suggestion: suggestion.map(|s| s.into()),
        }
    }

    pub fn exit_code(&self) -> ExitCode {
        self.category.exit_code()
    }

    /// Copy with `prefix` prepended to the message; details/suggestion kept.
    /// Lets store-level wrappers attach context (e.g. the key name) without
    /// burying the error under an opaque anyhow context layer.
    pub fn with_message_prefix(&self, prefix: &str) -> Self {
        Self {
            message: format!("{prefix}{}", self.message),
            category: self.category,
            details: self.details.clone(),
            suggestion: self.suggestion.clone(),
        }
    }
}

/// Exit code for an error chain at a command boundary: a wrapped [`CLIError`]
/// supplies its category (network → 2, auth → 3); a transport-level reqwest
/// failure (connect/timeout) anywhere in the chain maps to network; anything
/// else is a user error (1). Keeps the documented exit-code contract honest
/// without every call site re-classifying.
pub fn exit_code_for_error(err: &anyhow::Error) -> ExitCode {
    if let Some(cli) = err.chain().find_map(|c| c.downcast_ref::<CLIError>()) {
        return cli.exit_code();
    }
    let is_transport = err.chain().any(|cause| {
        cause
            .downcast_ref::<reqwest::Error>()
            .is_some_and(|e| e.is_connect() || e.is_timeout())
    });
    if is_transport {
        return ExitCode::NetworkError;
    }
    ExitCode::UserError
}

impl fmt::Display for CLIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(ref details) = self.details {
            write!(f, "\n  {}", details)?;
        }
        if let Some(ref suggestion) = self.suggestion {
            write!(f, "\n  Suggestion: {}", suggestion)?;
        }
        Ok(())
    }
}

impl std::error::Error for CLIError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_code_values() {
        assert_eq!(ExitCode::Success.code(), 0);
        assert_eq!(ExitCode::UserError.code(), 1);
        assert_eq!(ExitCode::NetworkError.code(), 2);
        assert_eq!(ExitCode::AuthError.code(), 3);
        assert_eq!(ExitCode::ToolExit(130).code(), 130);
    }

    #[test]
    fn test_cli_error_creation() {
        let err = CLIError::new(
            "test error",
            ErrorCategory::User,
            None::<String>,
            None::<String>,
        );
        assert_eq!(err.to_string(), "test error");
    }

    #[test]
    fn test_cli_error_with_details_and_suggestion() {
        let err = CLIError::new(
            "Key not found",
            ErrorCategory::User,
            Some("No key matching 'foo' was found"),
            Some("Run 'aivo keys' to see available keys"),
        );
        let display = err.to_string();
        assert!(display.contains("Key not found"));
        assert!(display.contains("No key matching 'foo' was found"));
        assert!(display.contains("Run 'aivo keys'"));
    }

    #[test]
    fn test_cli_error_with_actionable_suggestion() {
        let err = CLIError::new(
            "Failed to connect to OpenRouter",
            ErrorCategory::Network,
            Some("HTTP 403: Invalid API key"),
            Some("Check your key with `aivo keys cat <id>` or add a new key with `aivo keys add`"),
        );
        let display = err.to_string();
        assert!(display.contains("Failed to connect"));
        assert!(display.contains("403"));
        assert!(
            display.contains("aivo keys cat"),
            "Error should suggest the 'keys cat' command"
        );
        assert!(
            display.contains("aivo keys add"),
            "Error should suggest the 'keys add' command"
        );
    }

    #[test]
    fn test_cli_error_no_details_or_suggestion() {
        let err = CLIError::new(
            "Simple error",
            ErrorCategory::User,
            None::<String>,
            None::<String>,
        );
        let display = err.to_string();
        assert_eq!(display, "Simple error");
    }

    #[test]
    fn exit_code_for_error_maps_cli_error_categories() {
        for (category, expected) in [
            (ErrorCategory::User, ExitCode::UserError),
            (ErrorCategory::Network, ExitCode::NetworkError),
            (ErrorCategory::Auth, ExitCode::AuthError),
        ] {
            let err: anyhow::Error =
                CLIError::new("boom", category, None::<String>, None::<String>).into();
            assert_eq!(exit_code_for_error(&err), expected, "{category:?}");
        }
    }

    #[test]
    fn exit_code_for_error_survives_anyhow_context() {
        use anyhow::Context;
        let err: anyhow::Error = CLIError::new(
            "no key",
            ErrorCategory::Auth,
            None::<String>,
            None::<String>,
        )
        .into();
        let wrapped = Err::<(), _>(err).context("while launching").unwrap_err();
        assert_eq!(exit_code_for_error(&wrapped), ExitCode::AuthError);
    }

    #[test]
    fn exit_code_for_error_defaults_to_user_error() {
        let err = anyhow::anyhow!("something else");
        assert_eq!(exit_code_for_error(&err), ExitCode::UserError);
    }

    #[test]
    fn worse_of_picks_higher_severity() {
        use ExitCode::*;
        // Auth beats network beats user beats success, regardless of order.
        assert_eq!(Success.worse_of(NetworkError), NetworkError);
        assert_eq!(NetworkError.worse_of(Success), NetworkError);
        assert_eq!(NetworkError.worse_of(AuthError), AuthError);
        assert_eq!(AuthError.worse_of(NetworkError), AuthError);
        assert_eq!(UserError.worse_of(NetworkError), NetworkError);
        assert_eq!(Success.worse_of(Success), Success);
    }
}
