use aivo::cli::{Cli, Commands};
use clap::Parser;

/// Simulates the alias rewriting done in main.rs
fn rewrite_alias(args: &[&str]) -> Vec<String> {
    let aliases = ["claude", "codex", "gemini", "opencode", "pi"];
    let raw: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    if raw.len() > 1 && aliases.contains(&raw[1].as_str()) {
        let mut rewritten = vec![raw[0].clone(), "run".to_string()];
        rewritten.extend_from_slice(&raw[1..]);
        rewritten
    } else if raw.len() > 1 && raw[1] == "use" {
        let mut rewritten = vec![raw[0].clone(), "keys".to_string(), "use".to_string()];
        rewritten.extend_from_slice(&raw[2..]);
        rewritten
    } else {
        raw
    }
}

#[test]
fn no_command_yields_none() {
    let cli = Cli::try_parse_from(["aivo"]).unwrap();
    assert!(cli.command.is_none());
}

#[test]
fn version_flag() {
    let cli = Cli::try_parse_from(["aivo", "--version"]).unwrap();
    assert!(cli.version);
}

#[test]
fn help_flag() {
    let cli = Cli::try_parse_from(["aivo", "--help"]).unwrap();
    assert!(cli.help);
}

#[test]
fn run_with_tool() {
    let cli = Cli::try_parse_from(["aivo", "run", "claude"]).unwrap();
    if let Some(Commands::Run(args)) = cli.command {
        assert_eq!(args.tool, Some("claude".to_string()));
        assert!(args.debug.is_none());
        assert!(!args.dry_run);
    } else {
        panic!("Expected Run command");
    }
}

#[test]
fn run_without_tool() {
    let cli = Cli::try_parse_from(["aivo", "run"]).unwrap();
    if let Some(Commands::Run(args)) = cli.command {
        assert!(args.tool.is_none());
    } else {
        panic!("Expected Run command");
    }
}

#[test]
fn tool_alias_claude() {
    let args = rewrite_alias(&["aivo", "claude"]);
    let cli = Cli::try_parse_from(&args).unwrap();
    assert!(
        matches!(cli.command, Some(Commands::Run(ref a)) if a.tool == Some("claude".to_string()))
    );
}

#[test]
fn tool_alias_codex_with_model() {
    let args = rewrite_alias(&["aivo", "codex", "--model", "o4-mini"]);
    let cli = Cli::try_parse_from(&args).unwrap();
    if let Some(Commands::Run(run_args)) = cli.command {
        assert_eq!(run_args.tool, Some("codex".to_string()));
        assert_eq!(run_args.model, Some("o4-mini".to_string()));
    } else {
        panic!("Expected Run command");
    }
}

#[test]
fn tool_alias_gemini() {
    let args = rewrite_alias(&["aivo", "gemini", "--debug"]);
    let cli = Cli::try_parse_from(&args).unwrap();
    if let Some(Commands::Run(run_args)) = cli.command {
        assert_eq!(run_args.tool, Some("gemini".to_string()));
        assert_eq!(run_args.debug, Some(String::new()));
    } else {
        panic!("Expected Run command");
    }
}

#[test]
fn tool_alias_pi() {
    let args = rewrite_alias(&["aivo", "pi"]);
    let cli = Cli::try_parse_from(&args).unwrap();
    assert!(matches!(cli.command, Some(Commands::Run(ref a)) if a.tool == Some("pi".to_string())));
}

#[test]
fn tool_alias_opencode() {
    let args = rewrite_alias(&["aivo", "opencode"]);
    let cli = Cli::try_parse_from(&args).unwrap();
    assert!(
        matches!(cli.command, Some(Commands::Run(ref a)) if a.tool == Some("opencode".to_string()))
    );
}

#[test]
fn use_alias_rewrites_to_keys_use() {
    let args = rewrite_alias(&["aivo", "use", "my-key"]);
    let cli = Cli::try_parse_from(&args).unwrap();
    if let Some(Commands::Keys(keys_args)) = cli.command {
        assert_eq!(keys_args.action.as_deref(), Some("use"));
        assert_eq!(keys_args.args, vec!["my-key"]);
    } else {
        panic!("Expected Keys command");
    }
}

#[test]
fn keys_add_with_all_flags() {
    let cli = Cli::try_parse_from([
        "aivo",
        "keys",
        "add",
        "--name",
        "test-provider",
        "--base-url",
        "https://example.com/v1",
        "--key",
        "sk-test-123",
    ])
    .unwrap();
    if let Some(Commands::Keys(args)) = cli.command {
        assert_eq!(args.action.as_deref(), Some("add"));
        assert_eq!(args.name.as_deref(), Some("test-provider"));
        assert_eq!(args.base_url.as_deref(), Some("https://example.com/v1"));
        assert_eq!(args.key.as_deref(), Some("sk-test-123"));
    } else {
        panic!("Expected Keys command");
    }
}

#[test]
fn chat_execute_short_flag() {
    let cli = Cli::try_parse_from(["aivo", "chat", "-x", "hello world"]).unwrap();
    if let Some(Commands::Chat(args)) = cli.command {
        assert_eq!(args.execute, Some("hello world".to_string()));
    } else {
        panic!("Expected Chat command");
    }
}

#[test]
fn chat_model_key_and_execute() {
    let cli =
        Cli::try_parse_from(["aivo", "chat", "-k", "my-key", "-m", "gpt-4o", "-x", "hi"]).unwrap();
    if let Some(Commands::Chat(args)) = cli.command {
        assert_eq!(args.key, Some("my-key".to_string()));
        assert_eq!(args.model, Some("gpt-4o".to_string()));
        assert_eq!(args.execute, Some("hi".to_string()));
    } else {
        panic!("Expected Chat command");
    }
}

#[test]
fn chat_empty_model_triggers_picker() {
    let cli = Cli::try_parse_from(["aivo", "chat", "--model"]).unwrap();
    if let Some(Commands::Chat(args)) = cli.command {
        assert_eq!(args.model, Some(String::new()));
    } else {
        panic!("Expected Chat command");
    }
}

#[test]
fn chat_json_flag_parses() {
    let cli = Cli::try_parse_from(["aivo", "chat", "-x", "hi", "--json"]).unwrap();
    if let Some(Commands::Chat(args)) = cli.command {
        assert!(args.json);
    } else {
        panic!("Expected Chat command");
    }
}

#[test]
fn serve_default_port() {
    let cli = Cli::try_parse_from(["aivo", "serve"]).unwrap();
    if let Some(Commands::Serve(args)) = cli.command {
        assert_eq!(args.port, 24860);
    } else {
        panic!("Expected Serve command");
    }
}

#[test]
fn serve_custom_port() {
    let cli = Cli::try_parse_from(["aivo", "serve", "-p", "8080"]).unwrap();
    if let Some(Commands::Serve(args)) = cli.command {
        assert_eq!(args.port, 8080);
    } else {
        panic!("Expected Serve command");
    }
}

#[test]
fn models_search_flag() {
    let cli = Cli::try_parse_from(["aivo", "models", "-s", "sonnet"]).unwrap();
    if let Some(Commands::Models(args)) = cli.command {
        assert_eq!(args.search.as_deref(), Some("sonnet"));
        assert!(!args.refresh);
    } else {
        panic!("Expected Models command");
    }
}

#[test]
fn models_refresh_flag() {
    let cli = Cli::try_parse_from(["aivo", "models", "-r"]).unwrap();
    if let Some(Commands::Models(args)) = cli.command {
        assert!(args.refresh);
    } else {
        panic!("Expected Models command");
    }
}

#[test]
fn info_command() {
    let cli = Cli::try_parse_from(["aivo", "info"]).unwrap();
    if let Some(Commands::Info(args)) = cli.command {
        assert!(!args.ping);
    } else {
        panic!("Expected Info command");
    }
}

#[test]
fn info_command_with_ping() {
    let cli = Cli::try_parse_from(["aivo", "info", "--ping"]).unwrap();
    if let Some(Commands::Info(args)) = cli.command {
        assert!(args.ping);
    } else {
        panic!("Expected Info command");
    }
}

#[test]
fn ls_alias_for_info() {
    let cli = Cli::try_parse_from(["aivo", "ls"]).unwrap();
    if let Some(Commands::Info(args)) = cli.command {
        assert!(!args.ping);
    } else {
        panic!("Expected Info command via ls alias");
    }
}

#[test]
fn ls_alias_with_ping() {
    let cli = Cli::try_parse_from(["aivo", "ls", "--ping"]).unwrap();
    if let Some(Commands::Info(args)) = cli.command {
        assert!(args.ping);
    } else {
        panic!("Expected Info command via ls alias");
    }
}

#[test]
fn run_passthrough_args_after_separator() {
    let cli = Cli::try_parse_from(["aivo", "run", "claude", "--", "--some-flag", "value"]).unwrap();
    if let Some(Commands::Run(args)) = cli.command {
        assert!(args.args.contains(&"--some-flag".to_string()));
        assert!(args.args.contains(&"value".to_string()));
    } else {
        panic!("Expected Run command");
    }
}

#[test]
fn non_alias_not_rewritten() {
    let args = rewrite_alias(&["aivo", "keys"]);
    let cli = Cli::try_parse_from(&args).unwrap();
    assert!(matches!(cli.command, Some(Commands::Keys(_))));
}

#[test]
fn audio_with_prompt() {
    let cli = Cli::try_parse_from(["aivo", "audio", "hello world"]).unwrap();
    if let Some(Commands::Audio(args)) = cli.command {
        assert_eq!(args.prompt.as_deref(), Some("hello world"));
        assert!(args.file.is_none());
        assert!(!args.overwrite);
        assert!(!args.no_play);
    } else {
        panic!("Expected Audio command");
    }
}

#[test]
fn audio_with_file_short_flag() {
    let cli = Cli::try_parse_from(["aivo", "audio", "-f", "script.txt"]).unwrap();
    if let Some(Commands::Audio(args)) = cli.command {
        assert_eq!(args.file.as_deref(), Some("script.txt"));
        assert!(args.prompt.is_none());
    } else {
        panic!("Expected Audio command");
    }
}

#[test]
fn audio_with_file_long_flag() {
    let cli = Cli::try_parse_from(["aivo", "audio", "--file", "lines.md"]).unwrap();
    if let Some(Commands::Audio(args)) = cli.command {
        assert_eq!(args.file.as_deref(), Some("lines.md"));
    } else {
        panic!("Expected Audio command");
    }
}

#[test]
fn audio_with_file_short_flag_without_path_uses_stdin_marker() {
    let cli = Cli::try_parse_from(["aivo", "audio", "-f"]).unwrap();
    if let Some(Commands::Audio(args)) = cli.command {
        assert_eq!(args.file.as_deref(), Some("-"));
        assert!(args.prompt.is_none());
    } else {
        panic!("Expected Audio command");
    }
}

#[test]
fn audio_with_file_long_flag_without_path_uses_stdin_marker() {
    let cli = Cli::try_parse_from(["aivo", "audio", "--file"]).unwrap();
    if let Some(Commands::Audio(args)) = cli.command {
        assert_eq!(args.file.as_deref(), Some("-"));
    } else {
        panic!("Expected Audio command");
    }
}

#[test]
fn audio_prompt_and_file_are_mutually_exclusive() {
    let err = Cli::try_parse_from(["aivo", "audio", "hello", "-f", "script.txt"]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("cannot be used") || msg.contains("conflict"),
        "expected conflict error, got: {msg}"
    );
}

#[test]
fn audio_with_list_flag() {
    let cli = Cli::try_parse_from(["aivo", "audio", "--list"]).unwrap();
    if let Some(Commands::Audio(args)) = cli.command {
        assert!(args.list);
        assert!(args.prompt.is_none());
        assert!(args.file.is_none());
    } else {
        panic!("Expected Audio command");
    }
}

#[test]
fn audio_rejects_removed_history_flag() {
    let err = Cli::try_parse_from(["aivo", "audio", "--history"]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unexpected argument") || msg.contains("unrecognized"),
        "expected unknown flag error, got: {msg}"
    );
}

#[test]
fn audio_rejects_removed_restart_flag() {
    let err = Cli::try_parse_from(["aivo", "audio", "hello", "--restart"]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unexpected argument") || msg.contains("unrecognized"),
        "expected unknown flag error, got: {msg}"
    );
}

#[test]
fn audio_full_flag_set() {
    let cli = Cli::try_parse_from([
        "aivo",
        "audio",
        "narration",
        "-m",
        "tts-1-hd",
        "-k",
        "openai",
        "-o",
        "/tmp/out.mp3",
        "--voice",
        "nova",
        "--format",
        "wav",
        "--speed",
        "1.25",
        "--no-play",
        "--overwrite",
        "--json",
    ])
    .unwrap();
    if let Some(Commands::Audio(args)) = cli.command {
        assert_eq!(args.prompt.as_deref(), Some("narration"));
        assert_eq!(args.model.as_deref(), Some("tts-1-hd"));
        assert_eq!(args.key.as_deref(), Some("openai"));
        assert_eq!(args.output.as_deref(), Some("/tmp/out.mp3"));
        assert_eq!(args.voice.as_deref(), Some("nova"));
        assert_eq!(args.format.as_deref(), Some("wav"));
        assert_eq!(args.speed, Some(1.25));
        assert!(args.no_play);
        assert!(args.overwrite);
        assert!(args.json);
    } else {
        panic!("Expected Audio command");
    }
}

#[test]
fn track_command_is_unrecognized() {
    let err = Cli::try_parse_from(["aivo", "track", "hello"]).unwrap_err();
    let msg = err.to_string();
    // clap reports "unrecognized subcommand" or similar — the specific
    // wording depends on the version; just ensure parsing fails.
    assert!(
        msg.contains("track") || msg.contains("unrecognized") || msg.contains("subcommand"),
        "expected unrecognized-subcommand error, got: {msg}"
    );
}
