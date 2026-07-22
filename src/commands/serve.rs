//! ServeCommand — starts a local OpenAI-compatible HTTP server.

use anyhow::Result;
use std::collections::HashMap;
use std::net::IpAddr;

use crate::errors::ExitCode;
use crate::services::request_log::RequestLogger;
use crate::services::serve_router::{ServeRouter, ServeRouterConfig};
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

pub struct ServeParams {
    pub port: u16,
    pub host: String,
    pub key_override: Option<ApiKey>,
    pub log: Option<String>,
    pub failover_keys: Vec<ApiKey>,
    pub cors: bool,
    pub timeout: u64,
    pub auth_token: Option<String>,
    pub aliases: HashMap<String, String>,
}

pub struct ServeCommand {
    session_store: SessionStore,
}

impl ServeCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(&self, params: ServeParams) -> ExitCode {
        match self.execute_internal(params).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                crate::errors::exit_code_for_error(&e)
            }
        }
    }

    async fn execute_internal(&self, params: ServeParams) -> Result<ExitCode> {
        let ServeParams {
            port,
            host,
            key_override,
            log,
            failover_keys,
            cors,
            timeout,
            auth_token,
            aliases,
        } = params;

        // Resolve auth token: None → no auth, Some("") → generate, Some(t) → use as-is
        let auth_token = match auth_token {
            Some(ref t) if t.is_empty() => Some(crate::services::serve_router::random_auth_token()),
            other => other,
        };

        // `--cors` is not required for the risk — it just widens the attacker
        // pool from network-reachable hosts to any webpage the user visits.
        if auth_token.is_none() && host_binds_publicly(&host) {
            let amplifier = if cors {
                " with --cors, so any webpage the user visits can use it too"
            } else {
                ""
            };
            eprintln!(
                "{} serve is bound to {} with no --auth-token; anyone who can reach this host could use this key{}. Add --auth-token or bind to 127.0.0.1.",
                style::yellow("Warning:"),
                host,
                amplifier,
            );
        }
        let key = match key_override {
            Some(k) => k,
            None => {
                eprintln!(
                    "{} No API key configured. Run 'aivo keys add' first.",
                    style::red("Error:")
                );
                return Ok(ExitCode::AuthError);
            }
        };

        // Provider OAuth (grok/codex) is proxyable; CLI-bound OAuth isn't.
        if key.is_any_oauth() && !key.is_provider_oauth() {
            eprintln!(
                "{} Key '{}' is an OAuth credential — `aivo serve` can't proxy it.",
                style::red("Error:"),
                key.display_name()
            );
            eprintln!(
                "  {} Use `{}` to launch the tool directly, or switch to a regular API key with `aivo use`.",
                style::dim("hint:"),
                key.oauth_tool_hint()
            );
            return Ok(ExitCode::UserError);
        }

        if is_self_proxy_target(&key.base_url, port, &host) {
            anyhow::bail!(
                "Refusing to start `aivo serve`: active upstream {} points back to http://{}:{} and would proxy into itself. Switch to a real provider key with `aivo use <name>` or pass `--key <name>`.",
                key.base_url,
                host,
                port
            );
        }

        let grok_fallback = if key.is_grok_oauth() {
            crate::services::serve_router::resolve_grok_fallback(&self.session_store).await
        } else {
            None
        };
        let config = ServeRouterConfig::from_key(&key, cors, timeout, auth_token, aliases)
            .with_grok_fallback(grok_fallback);

        // Capture display info before moving key into the router
        let display_name = key.display_name().to_string();
        // Sentinel set in run.rs when the positional REF is an HF ref.
        let hf_mode = key.id == "aivo-hf-local";
        let display_host = if config.is_copilot {
            "github.com/copilot".to_string()
        } else if hf_mode {
            "local llama-server".to_string()
        } else {
            let stripped = key
                .base_url
                .strip_prefix("https://")
                .or_else(|| key.base_url.strip_prefix("http://"))
                .unwrap_or(&key.base_url);
            super::truncate_url_for_display(stripped, 40)
        };

        let logger = match log {
            Some(ref path) if !path.is_empty() => {
                RequestLogger::new_with_path(std::path::Path::new(path)).await
            }
            Some(_) => Some(RequestLogger::new_stdout()),
            None => None,
        };

        let failover_count = failover_keys.len();
        let log_display = logger.as_ref().map(|l| l.path_display().to_string());
        let auth_display = config.auth_token.clone();
        let router = ServeRouter::new(config, key, self.session_store.logs())
            .with_oauth_persist(self.session_store.clone())
            .with_logger(logger)
            .with_failover_keys(failover_keys);

        // Bind eagerly — errors here (e.g. "address already in use") before printing startup
        let (mut handle, shutdown) = router.start_background(&host, port).await?;

        // Format display URL — bracket IPv6 addresses
        let display_addr = if host.contains(':') {
            format!("[{}]", host)
        } else {
            host.clone()
        };
        eprintln!(
            "{} Listening on http://{}:{}",
            style::success_symbol(),
            display_addr,
            port
        );

        // Info line: key · host · log
        let mut info_parts = vec![display_name.clone()];
        // Only show base URL if it differs from the key name (avoid "ollama · ollama")
        if display_host != display_name {
            info_parts.push(style::dim(&display_host));
        }
        if let Some(ref path) = log_display {
            info_parts.push(format!("log: {}", style::dim(path)));
        }
        eprintln!("  {}", info_parts.join(" · "));

        if let Some(ref token) = auth_display {
            eprintln!("  auth: {}", style::dim(token));
        }
        if cors {
            eprintln!("  cors: {}", style::dim("enabled"));
        }
        if failover_count > 0 {
            eprintln!(
                "  {} failover: {} additional key{}",
                style::dim("↳"),
                failover_count,
                if failover_count == 1 { "" } else { "s" }
            );
        }
        if hf_mode {
            eprintln!(
                "  {} GET /v1/models on this URL to discover the model id",
                style::dim("·")
            );
            eprintln!(
                "  {} POST /v1/chat/completions accepts any value in the `model` field",
                style::dim("·")
            );
        }
        eprintln!("  {}", style::dim("Press Ctrl+C to stop"));

        // SIGTERM gets the same graceful shutdown as Ctrl-C. Without
        // this, `kill <aivo-pid>` from outside (or pkill, or a process
        // manager) would terminate aivo without running cleanup, leaking
        // any HF-mode llama-server child. On Windows `signal_terminate`
        // returns `()`; the lint about a unit binding is the cost of
        // sharing one code path.
        #[allow(clippy::let_unit_value)]
        let mut sigterm = signal_terminate()?;
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal?;
                eprintln!("\n  Shutting down...");
                shutdown.notify_one();
                wait_router_shutdown(&mut handle).await?;
            }
            _ = sigterm_recv(&mut sigterm) => {
                eprintln!("\n  Shutting down (SIGTERM)...");
                shutdown.notify_one();
                wait_router_shutdown(&mut handle).await?;
            }
            result = &mut handle => match result {
                Ok(Ok(())) => {
                    anyhow::bail!("serve router stopped unexpectedly");
                }
                Ok(Err(err)) => {
                    return Err(err);
                }
                Err(err) if err.is_cancelled() => {}
                Err(err) => {
                    anyhow::bail!("serve router task failed: {}", err);
                }
            },
        }

        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!("{} aivo serve [REF]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Start a local any-to-any AI proxy. Clients speak OpenAI Chat, Anthropic, Gemini, or the Responses API; aivo bridges to whatever protocol the active key's provider answers. Serves a local llama-server when given a hf:/URL REF."
            )
        );
        println!();
        println!("{}", style::bold("Endpoints:"));
        let print_ep = |path: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<44}", path)),
                style::dim(desc)
            );
        };
        print_ep("POST /v1/chat/completions", "OpenAI Chat Completions");
        print_ep("POST /v1/responses", "OpenAI Responses API");
        print_ep("POST /v1/messages", "Anthropic Messages");
        print_ep(
            "POST /v1beta/models/{model}:generateContent",
            "Gemini (+ :streamGenerateContent)",
        );
        print_ep("POST /v1/embeddings", "OpenAI Embeddings");
        print_ep("GET  /v1/models", "Model list from the upstream");
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt("-p, --port <PORT>", "Port to listen on (default: 24860)");
        print_opt(
            "--host <ADDR>",
            "Host address to bind to (default: 127.0.0.1)",
        );
        print_opt("-k, --key <id|name>", "Select API key by ID or name");
        print_opt("--log [PATH]", "Log requests as JSONL (stdout, or to PATH)");
        print_opt("--failover", "Enable multi-key failover on 429/5xx errors");
        print_opt("--cors", "Enable CORS headers for browser-based clients");
        print_opt(
            "--timeout <SECS>",
            "Upstream timeout in seconds (default: 300)",
        );
        print_opt(
            "--auth-token [TOKEN]",
            "Require a token: Bearer/x-api-key/x-goog-api-key/?key= (auto-generated if omitted)",
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo serve"));
        println!("  {}", style::dim("aivo serve --host 0.0.0.0 -p 8080"));
        println!(
            "  {}",
            style::dim("aivo serve hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        );
    }
}

/// Drains the router's join handle after a shutdown notify, with a 6s
/// grace period before forced abort.
async fn wait_router_shutdown(handle: &mut tokio::task::JoinHandle<Result<()>>) -> Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(6), &mut *handle).await {
        Ok(Ok(Ok(()))) => Ok(()),
        Ok(Ok(Err(err))) => Err(err),
        Ok(Err(err)) if err.is_cancelled() => Ok(()),
        Ok(Err(err)) => anyhow::bail!("serve router task failed: {}", err),
        Err(_) => {
            eprintln!("  Grace period expired, aborting");
            handle.abort();
            let _ = handle.await;
            Ok(())
        }
    }
}

/// Installs a SIGTERM listener on Unix. On Windows this is a no-op
/// since the platform lacks SIGTERM; SIGINT (Ctrl-C) is handled separately.
#[cfg(unix)]
fn signal_terminate() -> Result<tokio::signal::unix::Signal> {
    use tokio::signal::unix::{SignalKind, signal};
    Ok(signal(SignalKind::terminate())?)
}

#[cfg(not(unix))]
fn signal_terminate() -> Result<()> {
    Ok(())
}

#[cfg(unix)]
async fn sigterm_recv(sig: &mut tokio::signal::unix::Signal) {
    sig.recv().await;
}

#[cfg(not(unix))]
async fn sigterm_recv(_: &mut ()) {
    // No SIGTERM on Windows; future never resolves.
    std::future::pending::<()>().await
}

/// True when the bind host reaches beyond loopback.
fn host_binds_publicly(bind_host: &str) -> bool {
    if bind_host == "0.0.0.0" || bind_host == "::" {
        return true;
    }
    match bind_host.trim_matches(['[', ']']).parse::<IpAddr>() {
        Ok(ip) => !ip.is_loopback(),
        Err(_) => !bind_host.eq_ignore_ascii_case("localhost"),
    }
}

fn is_self_proxy_target(base_url: &str, port: u16, bind_host: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return false;
    };

    let Some(host) = url.host_str() else {
        return false;
    };
    let Some(target_port) = url.port_or_known_default() else {
        return false;
    };

    if target_port != port {
        return false;
    }

    // When binding to 0.0.0.0, all loopback addresses are self-proxy targets
    let binds_all = bind_host == "0.0.0.0" || bind_host == "::";

    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }

    host.trim_matches(['[', ']'])
        .parse::<IpAddr>()
        .is_ok_and(|ip| {
            if binds_all {
                // When bound to all interfaces, any local address is a self-proxy target
                ip.is_loopback() || ip.is_unspecified()
            } else {
                ip.is_loopback()
            }
        })
}

#[cfg(test)]
mod tests {
    use super::{host_binds_publicly, is_self_proxy_target};

    #[test]
    fn host_binds_publicly_recognizes_loopback_and_non_loopback() {
        assert!(host_binds_publicly("0.0.0.0"));
        assert!(host_binds_publicly("::"));
        assert!(host_binds_publicly("192.168.1.10"));
        assert!(host_binds_publicly("example.com"));

        assert!(!host_binds_publicly("127.0.0.1"));
        assert!(!host_binds_publicly("::1"));
        assert!(!host_binds_publicly("[::1]"));
        assert!(!host_binds_publicly("localhost"));
        assert!(!host_binds_publicly("LOCALHOST"));
    }

    #[test]
    fn detects_localhost_self_proxy() {
        assert!(is_self_proxy_target(
            "http://127.0.0.1:24860",
            24860,
            "127.0.0.1"
        ));
        assert!(is_self_proxy_target(
            "http://127.0.0.1:24860/v1",
            24860,
            "127.0.0.1"
        ));
        assert!(is_self_proxy_target(
            "http://localhost:24860",
            24860,
            "127.0.0.1"
        ));
        assert!(is_self_proxy_target(
            "http://[::1]:24860/v1",
            24860,
            "127.0.0.1"
        ));
    }

    #[test]
    fn ignores_other_ports_and_hosts() {
        assert!(!is_self_proxy_target(
            "http://127.0.0.1:8080",
            24860,
            "127.0.0.1"
        ));
        assert!(!is_self_proxy_target(
            "https://api.openai.com/v1",
            24860,
            "127.0.0.1"
        ));
        assert!(!is_self_proxy_target("not-a-url", 24860, "127.0.0.1"));
    }

    #[test]
    fn detects_self_proxy_when_bound_to_all() {
        assert!(is_self_proxy_target(
            "http://127.0.0.1:24860",
            24860,
            "0.0.0.0"
        ));
        assert!(is_self_proxy_target(
            "http://localhost:24860",
            24860,
            "0.0.0.0"
        ));
    }

    #[test]
    fn host_binds_publicly_handles_127_subnet_loopback() {
        // The whole 127.0.0.0/8 range is loopback per RFC; std's IpAddr::is_loopback
        // recognises this. Any address in this range should NOT trigger the warning.
        for ip in ["127.0.0.1", "127.0.0.2", "127.0.0.255", "127.255.255.254"] {
            assert!(
                !host_binds_publicly(ip),
                "loopback /8 host {ip} should not trigger public-bind warning",
            );
        }
    }

    #[test]
    fn host_binds_publicly_treats_unparseable_non_localhost_as_public() {
        // A hostname that isn't "localhost" and doesn't parse as an IP is
        // treated as public — the warning is best-effort, false positives
        // are safer than false negatives.
        assert!(host_binds_publicly("api.internal.example"));
        assert!(host_binds_publicly("server-1"));
        assert!(host_binds_publicly("[::ffff:8.8.8.8]"));
    }

    #[test]
    fn self_proxy_detects_loopback_v6_link_local_when_bound_to_all() {
        // `::1` is the canonical IPv6 loopback; verifying both bracketed
        // and unbracketed forms work alongside is_self_proxy_target's
        // bracket-stripping.
        assert!(is_self_proxy_target("http://[::1]:9000/", 9000, "0.0.0.0"));
        assert!(is_self_proxy_target("http://[::1]:9000/", 9000, "::"));
    }

    #[test]
    fn self_proxy_does_not_match_external_hosts() {
        // Sanity: a real public hostname must never be classified as
        // self-proxy regardless of bind host.
        for bind in ["127.0.0.1", "0.0.0.0", "::"] {
            assert!(
                !is_self_proxy_target("https://api.openai.com/v1", 9000, bind),
                "external API must not be self-proxy when bind={bind}",
            );
        }
    }

    #[test]
    fn self_proxy_ignores_url_path_only() {
        // Paths must not influence classification — only host+port.
        assert!(is_self_proxy_target(
            "http://127.0.0.1:24860/v1/chat/completions",
            24860,
            "127.0.0.1"
        ));
        assert!(!is_self_proxy_target(
            "http://127.0.0.1:8080/v1/chat/completions",
            24860,
            "127.0.0.1"
        ));
    }
}
