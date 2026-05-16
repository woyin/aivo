//! Custom DNS resolver for Termux. aivo's release binary is static-musl, so
//! its libc resolver reads `/etc/resolv.conf` literally — a file Termux does
//! not populate (Android's read-only `/etc` is bypassed by bionic via netd).
//! Result: `getaddrinfo` returns `EAI_AGAIN` for every lookup. We plug a
//! hickory-resolver into reqwest that reads `$PREFIX/etc/resolv.conf` (the
//! Termux-prefixed path) and falls back to Cloudflare + Google.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, OnceLock};

use hickory_resolver::Resolver;
use hickory_resolver::config::{CLOUDFLARE, GOOGLE, NameServerConfig, ResolverConfig};
use hickory_resolver::net::runtime::TokioRuntimeProvider;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};

/// Process-wide resolver. Lazily built on first call.
static TERMUX_RESOLVER: OnceLock<Arc<TermuxDnsResolver>> = OnceLock::new();

pub fn termux_dns_resolver() -> Arc<TermuxDnsResolver> {
    TERMUX_RESOLVER
        .get_or_init(|| Arc::new(TermuxDnsResolver::new()))
        .clone()
}

pub struct TermuxDnsResolver {
    inner: Resolver<TokioRuntimeProvider>,
}

impl TermuxDnsResolver {
    fn new() -> Self {
        let config = build_resolver_config();
        let inner = Resolver::builder_with_config(config, TokioRuntimeProvider::default())
            .build()
            .expect("hickory resolver build");
        Self { inner }
    }
}

impl Resolve for TermuxDnsResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_owned();
        if let Some(addrs) = literal_lookup(&host) {
            return Box::pin(async move { Ok(addrs) });
        }
        let resolver = self.inner.clone();
        Box::pin(async move {
            let lookup = resolver.lookup_ip(host).await?;
            let ips: Vec<SocketAddr> = lookup.iter().map(|ip| SocketAddr::new(ip, 0)).collect();
            let addrs: Addrs = Box::new(ips.into_iter());
            Ok(addrs)
        })
    }
}

/// Short-circuit names a real DNS lookup would either fail on or waste a
/// round-trip resolving.
fn literal_lookup(host: &str) -> Option<Addrs> {
    if host.eq_ignore_ascii_case("localhost") {
        let ips: Vec<SocketAddr> = vec![
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 0),
        ];
        return Some(Box::new(ips.into_iter()));
    }
    None
}

fn build_resolver_config() -> ResolverConfig {
    if let Some(cfg) = config_from_termux_resolv_conf() {
        return cfg;
    }
    let mut cfg = ResolverConfig::udp_and_tcp(&CLOUDFLARE);
    for ns in ResolverConfig::udp_and_tcp(&GOOGLE).name_servers.iter() {
        cfg.add_name_server(ns.clone());
    }
    cfg
}

fn config_from_termux_resolv_conf() -> Option<ResolverConfig> {
    let prefix = std::env::var("PREFIX").ok()?;
    let path = std::path::Path::new(&prefix).join("etc/resolv.conf");
    let text = std::fs::read_to_string(&path).ok()?;
    config_from_resolv_conf_text(&text)
}

fn config_from_resolv_conf_text(text: &str) -> Option<ResolverConfig> {
    let ips = parse_nameservers(text);
    if ips.is_empty() {
        return None;
    }
    let name_servers = ips.into_iter().map(NameServerConfig::udp_and_tcp).collect();
    Some(ResolverConfig::from_parts(None, vec![], name_servers))
}

fn parse_nameservers(text: &str) -> Vec<IpAddr> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some(rest) = line.strip_prefix("nameserver") else {
            continue;
        };
        let rest = rest.trim();
        if let Ok(ip) = rest.parse::<IpAddr>() {
            out.push(ip);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_termux_style_resolv_conf() {
        let text = "nameserver 8.8.8.8\nnameserver 8.8.4.4\n";
        assert_eq!(
            parse_nameservers(text),
            vec![
                IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
                IpAddr::V4(Ipv4Addr::new(8, 8, 4, 4)),
            ]
        );
    }

    #[test]
    fn skips_comments_and_unrelated_directives() {
        let text = "\
# comment
search example.com
nameserver 1.1.1.1  # inline
options ndots:2
nameserver 2606:4700:4700::1111
";
        assert_eq!(
            parse_nameservers(text),
            vec![
                IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)),
                IpAddr::V6("2606:4700:4700::1111".parse().unwrap()),
            ]
        );
    }

    #[test]
    fn ignores_malformed_nameserver_lines() {
        let text = "nameserver not-an-ip\nnameserver \nnameserver 999.999.999.999\n";
        assert!(parse_nameservers(text).is_empty());
    }

    #[test]
    fn config_from_text_returns_none_when_no_nameservers() {
        assert!(config_from_resolv_conf_text("search example.com\n").is_none());
    }

    #[test]
    fn config_from_text_collects_each_nameserver() {
        let cfg = config_from_resolv_conf_text("nameserver 1.1.1.1\nnameserver 8.8.8.8\n").unwrap();
        let ips: Vec<IpAddr> = cfg.name_servers.iter().map(|ns| ns.ip).collect();
        assert_eq!(
            ips,
            vec![
                IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)),
                IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            ]
        );
    }

    #[test]
    fn fallback_config_includes_cloudflare_and_google() {
        let cfg = build_resolver_config();
        let ips: Vec<IpAddr> = cfg.name_servers.iter().map(|ns| ns.ip).collect();
        // At minimum we must see one Cloudflare + one Google v4 in the
        // fallback so a default install has something to query.
        assert!(ips.contains(&IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
        assert!(ips.contains(&IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
    }

    #[test]
    fn localhost_short_circuits() {
        let addrs = literal_lookup("localhost").expect("localhost addrs");
        let collected: Vec<SocketAddr> = addrs.collect();
        assert!(collected.iter().any(|a| a.ip() == Ipv4Addr::LOCALHOST));
        assert!(collected.iter().any(|a| a.ip() == Ipv6Addr::LOCALHOST));
    }

    #[test]
    fn non_localhost_returns_none() {
        assert!(literal_lookup("example.com").is_none());
    }
}
