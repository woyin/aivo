//! ServeCommand — starts a local OpenAI-compatible HTTP server.

use anyhow::Result;
use std::net::IpAddr;

use crate::errors::ExitCode;
use crate::services::log_store::LogStore;
use crate::services::provider_profile::provider_profile_for_key;
use crate::services::request_log::RequestLogger;
use crate::services::serve_router::{ServeRouter, ServeRouterConfig};
use crate::services::session_store::ApiKey;
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
}

pub struct ServeCommand {
    log_store: LogStore,
}

impl Default for ServeCommand {
    fn default() -> Self {
        Self::new(LogStore::new(std::path::PathBuf::from(".config/aivo")))
    }
}

impl ServeCommand {
    pub fn new(log_store: LogStore) -> Self {
        Self { log_store }
    }

    pub async fn execute(&self, params: ServeParams) -> ExitCode {
        match self.execute_internal(params).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
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
        } = params;

        // Resolve auth token: None → no auth, Some("") → generate, Some(t) → use as-is
        let auth_token = match auth_token {
            Some(ref t) if t.is_empty() => {
                use rand::Rng;
                let token: String = rand::thread_rng()
                    .sample_iter(&rand::distributions::Alphanumeric)
                    .take(32)
                    .map(char::from)
                    .collect();
                Some(token)
            }
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

        let profile = provider_profile_for_key(&key);
        let is_copilot = profile.serve_flags.is_copilot;
        let is_openrouter = profile.serve_flags.is_openrouter;
        let upstream_protocol = profile.default_protocol;

        if is_self_proxy_target(&key.base_url, port, &host) {
            anyhow::bail!(
                "Refusing to start `aivo serve`: active upstream {} points back to http://{}:{} and would proxy into itself. Switch to a real provider key with `aivo use <name>` or pass `--key <name>`.",
                key.base_url,
                host,
                port
            );
        }

        // Capture display info before moving key into the router
        let display_name = key.display_name().to_string();
        let display_host = if is_copilot {
            "github.com/copilot".to_string()
        } else {
            let stripped = key
                .base_url
                .strip_prefix("https://")
                .or_else(|| key.base_url.strip_prefix("http://"))
                .unwrap_or(&key.base_url);
            super::truncate_url_for_display(stripped, 40)
        };

        let config = ServeRouterConfig {
            upstream_base_url: crate::services::provider_profile::resolve_starter_base_url(
                &key.base_url,
            ),
            upstream_api_key: key.key.as_str().to_string(),
            upstream_protocol,
            is_copilot,
            is_openrouter,
            is_starter: profile.serve_flags.is_starter,
            cors,
            timeout,
            auth_token,
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
        let router = ServeRouter::new(config, key, self.log_store.clone())
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
        eprintln!("  {}", style::dim("Press Ctrl+C to stop"));

        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal?;
                eprintln!("\n  Shutting down...");
                shutdown.notify_one();
                match tokio::time::timeout(std::time::Duration::from_secs(6), &mut handle).await {
                    Ok(Ok(Ok(()))) => {}
                    Ok(Ok(Err(err))) => return Err(err),
                    Ok(Err(err)) if err.is_cancelled() => {}
                    Ok(Err(err)) => anyhow::bail!("serve router task failed: {}", err),
                    Err(_) => {
                        eprintln!("  Grace period expired, aborting");
                        handle.abort();
                        let _ = handle.await;
                    }
                }
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
        println!("{} aivo serve", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Start a local OpenAI-compatible server that proxies to the active provider."
            )
        );
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
        print_opt(
            "-k, --key <id|name>",
            "Select API key by ID or name (-k opens key picker)",
        );
        print_opt(
            "--log [PATH]",
            "Log requests as JSONL (stdout, or to file if PATH given)",
        );
        print_opt("--failover", "Enable multi-key failover on 429/5xx errors");
        print_opt("--cors", "Enable CORS headers for browser-based clients");
        print_opt(
            "--timeout <SECS>",
            "Upstream timeout in seconds (default: 300)",
        );
        print_opt(
            "--auth-token [TOKEN]",
            "Require bearer token (auto-generated if no value given)",
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo serve"));
        println!("  {}", style::dim("aivo serve --host 0.0.0.0 -p 8080"));
        println!("  {}", style::dim("aivo serve -k openrouter"));
        println!("  {}", style::dim("aivo serve --log | jq ."));
        println!("  {}", style::dim("aivo serve --log /tmp/requests.jsonl"));
        println!("  {}", style::dim("aivo serve --cors --timeout 60"));
    }
}

/// True when the bind host reaches beyond loopback — i.e. any network
/// interface or an explicit non-loopback address. Used to gate warnings
/// that only matter when the server is actually exposed.
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
}
