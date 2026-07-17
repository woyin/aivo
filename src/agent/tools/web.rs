//! Web tools: fetch (SSRF-guarded) and gateway search, plus HTML-to-text.

use super::*;

/// Frame externally-fetched content (web/search/MCP) so the model treats it as data,
/// not instructions. The system prompt names this delimiter.
pub(crate) fn wrap_untrusted(source: &str, body: &str) -> String {
    format!("<untrusted source={source:?}>\n{body}\n</untrusted>")
}

pub(super) async fn web_fetch(args: &Value) -> Result<String, String> {
    let url = arg_str(args, "url")?;
    let max_chars = web_fetch_max_chars(args);
    let allow_local = web_fetch_allow_local();
    // Follow redirects manually (Policy::none) so every hop — the initial URL and
    // each 30x target — is re-validated against the SSRF blocklist below. The
    // default reqwest policy would chase a redirect into a private/loopback
    // address unchecked, which is the whole SSRF vector we're closing.
    let build_client = |pin: Option<(&str, &[SocketAddr])>| -> Result<reqwest::Client, String> {
        let mut b = reqwest::Client::builder()
            .timeout(Duration::from_secs(WEB_FETCH_TIMEOUT))
            .user_agent("aivo-agent/1.0")
            .redirect(reqwest::redirect::Policy::none());
        if let Some((host, addrs)) = pin {
            b = b.resolve_to_addrs(host, addrs);
        }
        b.build().map_err(|e| format!("build http client: {e}"))
    };

    let mut current = parse_http_url(url)?;
    let resp = {
        let mut hops = 0usize;
        loop {
            // Pin the vetted IPs so reqwest can't re-resolve to a private one between
            // check and connect (DNS-rebinding TOCTOU); `allow_local` opts out.
            let client = if allow_local {
                build_client(None)?
            } else {
                let host = current
                    .host_str()
                    .ok_or_else(|| format!("url has no host: {current}"))?;
                let addrs = guard_fetch_target(&current).await?;
                build_client(Some((host, &addrs)))?
            };
            let resp = client
                .get(current.clone())
                .send()
                .await
                .map_err(|e| fetch_failed(current.as_str(), &e.to_string()))?;
            if !resp.status().is_redirection() {
                break resp;
            }
            hops += 1;
            if hops > WEB_FETCH_MAX_REDIRECTS {
                return Err(format!(
                    "fetch {url}: too many redirects (>{WEB_FETCH_MAX_REDIRECTS})"
                ));
            }
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("fetch {current}: redirect without a Location header"))?;
            current = current
                .join(location)
                .map_err(|e| format!("bad redirect target {location:?}: {e}"))?;
            if !matches!(current.scheme(), "http" | "https") {
                return Err(format!(
                    "refusing to follow redirect to a non-http(s) URL: {current}"
                ));
            }
        }
    };
    let status = resp.status();
    let is_html = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|c| c.to_ascii_lowercase().contains("html"))
        .unwrap_or(false);
    // Stream the body and stop at the cap, so a giant (or hostile) response can't
    // buffer gigabytes into memory before we'd truncate it — `resp.bytes()` would
    // read the whole thing first.
    let body = read_capped(resp.bytes_stream(), WEB_FETCH_MAX_BYTES)
        .await
        .map_err(|e| format!("read body from {current}: {e}"))?;
    let raw = String::from_utf8_lossy(&body);
    let text = if is_html || raw.trim_start().starts_with('<') {
        html_to_text(&raw)
    } else {
        raw.into_owned()
    };
    let text: String = text.chars().take(max_chars).collect();
    if !status.is_success() {
        return Err(fetch_failed(current.as_str(), &format!("HTTP {status}")));
    }
    if text.trim().is_empty() {
        return Ok("(empty response)".to_string());
    }
    Ok(wrap_untrusted(&format!("web_fetch {current}"), &text))
}

// --- web_search: hosted /v1/search (layer B) ---

pub(super) struct SearchHit {
    pub(super) title: String,
    pub(super) url: String,
    pub(super) snippet: String,
}

/// `AIVO_SEARCH_ENDPOINT` overrides the gateway default (local wrangler in dev).
pub(super) fn search_endpoint() -> String {
    std::env::var("AIVO_SEARCH_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("{}/v1/search", crate::constants::AIVO_STARTER_REAL_URL))
}

/// Latched once search is known-exhausted this session (quota/auth/config), so
/// later web_search calls short-circuit instead of re-hitting the gateway.
pub(super) static SEARCH_EXHAUSTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(super) async fn web_search(args: &Value) -> Result<String, String> {
    use std::sync::atomic::Ordering::Relaxed;
    let query = arg_str(args, "query")?.trim();
    if query.is_empty() {
        return Err("web_search: empty query".to_string());
    }
    if SEARCH_EXHAUSTED.load(Relaxed) {
        return Err(search_exhausted(
            "Web search is unavailable for the rest of this session",
        ));
    }
    let max_results = arg_u64(args, "max_results")
        .map(|n| n as usize)
        .unwrap_or(WEB_SEARCH_DEFAULT_RESULTS)
        .clamp(1, WEB_SEARCH_MAX_RESULTS);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(WEB_FETCH_TIMEOUT))
        .build()
        .map_err(|e| format!("build http client: {e}"))?;
    // Device-signed (same auth as chat); the gateway holds the keys + quota.
    let builder = client
        .post(search_endpoint())
        .json(&json!({ "query": query, "max_results": max_results }));
    let resp = crate::services::device_fingerprint::with_starter_headers(builder)
        .send()
        .await
        .map_err(|e| search_unavailable(&format!("couldn't reach web search ({e})")))?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if status.is_success() {
        let hits = parse_search_results(&text, max_results);
        if hits.is_empty() {
            return Ok(format!("No web results for {query:?}."));
        }
        return Ok(wrap_untrusted(
            "web_search",
            &render_search_results(query, &hits),
        ));
    }
    let (message, latch) = classify_search_error(status.as_u16());
    if latch {
        // Persistent (quota/auth/config) — don't keep hammering the gateway.
        SEARCH_EXHAUSTED.store(true, Relaxed);
    }
    Err(message)
}

pub(super) fn parse_search_results(body: &str, max: usize) -> Vec<SearchHit> {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let Some(arr) = v.get("results").and_then(|r| r.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .take(max)
        .filter_map(|it| {
            let title = it.get("title")?.as_str()?.trim().to_string();
            let url = it.get("url")?.as_str()?.trim().to_string();
            if title.is_empty() || url.is_empty() {
                return None;
            }
            Some(SearchHit {
                title,
                url,
                snippet: it
                    .get("snippet")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string(),
            })
        })
        .collect()
}

/// Non-200 → (actionable layer-C message, whether to latch the session
/// exhausted). 401/429/503 are persistent (latch + tell the model to stop); 502
/// and network errors are transient (no latch). Leads with the human-readable
/// status so the truncated tool-card line stays meaningful.
pub(super) fn classify_search_error(status: u16) -> (String, bool) {
    match status {
        401 => (
            search_exhausted("Web search needs sign-in — run `aivo login`"),
            true,
        ),
        429 => (
            search_exhausted("Today's web-search quota is used up"),
            true,
        ),
        503 => (search_exhausted("Web search isn't configured"), true),
        502 => (search_unavailable("Web search is temporarily down"), false),
        _ => (
            search_unavailable(&format!("Web search failed (HTTP {status})")),
            false,
        ),
    }
}

/// Persistent unavailability — tell the model to STOP retrying (the engine also
/// short-circuits later calls via `SEARCH_EXHAUSTED`).
pub(super) fn search_exhausted(reason: &str) -> String {
    format!(
        "{reason}. Don't call web_search again this session — answer from what you already \
know (say plainly you couldn't search) or web_fetch a known URL; don't invent results."
    )
}

/// Transient unavailability — the model may proceed without search, but a later
/// call could succeed, so no "stop" steer.
pub(super) fn search_unavailable(reason: &str) -> String {
    format!(
        "{reason}. Answer from what you already know or web_fetch a known URL — don't \
invent search results, URLs, or facts."
    )
}

/// web_fetch failure → steer the model to its search content, not fabrication.
pub(super) fn fetch_failed(url: &str, reason: &str) -> String {
    format!(
        "Couldn't fetch {url} ({reason}) — the page may be unreachable from here or down. \
Answer from the web_search results you already have, or try a different result URL — do \
NOT invent this page's contents."
    )
}

pub(super) fn render_search_results(query: &str, hits: &[SearchHit]) -> String {
    let mut out = format!("Web search results for {query:?}:\n");
    for (i, h) in hits.iter().enumerate() {
        out.push_str(&format!("\n{}. {}\n   {}\n", i + 1, h.title, h.url));
        if !h.snippet.is_empty() {
            out.push_str("   ");
            out.push_str(&h.snippet);
            out.push('\n');
        }
    }
    out.push_str("\nUse web_fetch on a result URL to read the full page.");
    out
}

/// `AIVO_WEB_FETCH_ALLOW_LOCAL=1` opts back into fetching loopback/private hosts
/// (e.g. a local dev server you want the agent to read). Off by default so a
/// model — possibly steered by a prompt-injected page — can't turn `web_fetch`
/// into an SSRF against cloud metadata or internal services.
pub(super) fn web_fetch_allow_local() -> bool {
    std::env::var("AIVO_WEB_FETCH_ALLOW_LOCAL")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Parse a fetch URL, requiring an http(s) scheme.
pub(super) fn parse_http_url(raw: &str) -> Result<url::Url, String> {
    let u = url::Url::parse(raw).map_err(|e| format!("invalid url {raw:?}: {e}"))?;
    match u.scheme() {
        "http" | "https" => Ok(u),
        other => Err(format!("url must be http:// or https:// (got {other}://)")),
    }
}

/// SSRF guard: reject the host if ANY resolved address is non-public. Returns the vetted
/// addresses so the caller pins the connection to them (defeating a rebinding re-resolve).
pub(super) async fn guard_fetch_target(u: &url::Url) -> Result<Vec<SocketAddr>, String> {
    let host = u
        .host_str()
        .ok_or_else(|| format!("url has no host: {u}"))?;
    let port = u.port_or_known_default().unwrap_or(0);
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| format!("resolve {host}: {e}"))?
        .collect();
    if addrs.is_empty() {
        return Err(format!("resolve {host}: no addresses"));
    }
    for addr in &addrs {
        if ip_is_blocked(addr.ip()) {
            return Err(format!(
                "refusing to fetch {host}: resolves to a private/loopback address ({}). \
Set AIVO_WEB_FETCH_ALLOW_LOCAL=1 to allow local targets.",
                addr.ip()
            ));
        }
    }
    Ok(addrs)
}

/// Whether `ip` is in a range an outbound agent fetch must not reach: loopback,
/// RFC1918 private, link-local (includes the 169.254.169.254 cloud-metadata IP),
/// CGNAT, the unspecified/broadcast edges, IPv6 ULA/link-local, and the
/// IPv4-mapped/compatible forms of all of the above.
pub(super) fn ip_is_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || o[0] == 0
                || (o[0] == 100 && (o[1] & 0xc0) == 64) // 100.64.0.0/10 CGNAT
        }
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped().or_else(|| v6.to_ipv4()) {
                return ip_is_blocked(IpAddr::V4(mapped));
            }
            v6.is_loopback() || v6.is_unspecified() || ipv6_is_ula(v6) || ipv6_is_link_local(v6)
        }
    }
}

pub(super) fn ipv6_is_ula(ip: Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xfe00) == 0xfc00 // fc00::/7
}

pub(super) fn ipv6_is_link_local(ip: Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xffc0) == 0xfe80 // fe80::/10
}

/// Read a byte stream into a buffer, stopping once `max_bytes` is reached (the
/// final chunk is sliced, never over-collected). Bounds memory regardless of the
/// declared or actual body size. Generic over the chunk/error types so it's unit-
/// testable with a synthetic stream (no network).
pub(super) async fn read_capped<S, B, E>(mut stream: S, max_bytes: usize) -> Result<Vec<u8>, E>
where
    S: futures::Stream<Item = Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
{
    use futures::StreamExt;
    let mut body: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let remaining = max_bytes.saturating_sub(body.len());
        let bytes = chunk.as_ref();
        body.extend_from_slice(&bytes[..bytes.len().min(remaining)]);
        // Stop as soon as the cap is reached, rather than pulling (and discarding)
        // one more chunk from the network on the next iteration.
        if body.len() >= max_bytes {
            break;
        }
    }
    Ok(body)
}

/// Reduce HTML to readable text: drop `<script>/<style>/<head>` content, strip
/// tags (inserting newlines at block boundaries), decode the common entities,
/// and collapse whitespace. Best-effort — not a real HTML parser, but enough to
/// turn a page into something a model can read.
pub(super) fn html_to_text(html: &str) -> String {
    const BLOCKS: &[&str] = &[
        "p", "div", "br", "li", "tr", "section", "article", "header", "footer", "h1", "h2", "h3",
        "h4", "h5", "h6",
    ];
    let mut out = String::new();
    let mut rest = html;
    while let Some(lt) = rest.find('<') {
        out.push_str(&rest[..lt]);
        rest = &rest[lt..];
        let Some(gt) = rest.find('>') else {
            rest = ""; // unterminated tag — drop the remainder
            break;
        };
        let tag = &rest[1..gt];
        let is_close = tag.starts_with('/');
        let tname: String = tag
            .trim_start_matches('/')
            .split(|c: char| c.is_whitespace() || c == '/')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        rest = &rest[gt + 1..];
        if !is_close
            && !tag.ends_with('/')
            && matches!(
                tname.as_str(),
                "script" | "style" | "head" | "noscript" | "svg"
            )
        {
            // Skip the block's raw body up to its literal closing tag: its content
            // may hold `<`/`>` (e.g. `a < b` in a script) that must not be parsed
            // as markup, which would swallow the real `</script>`.
            let close = format!("</{tname}");
            rest = match find_ci(rest, &close) {
                Some(pos) => match rest[pos..].find('>') {
                    Some(g) => &rest[pos + g + 1..],
                    None => "",
                },
                None => "",
            };
        } else if BLOCKS.contains(&tname.as_str()) {
            out.push('\n');
        }
    }
    out.push_str(rest);
    collapse_whitespace(&decode_entities(&out))
}

/// Case-insensitive ASCII substring search, returning the byte offset in
/// `haystack`. Allocation-free (see body).
pub(super) fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    let needle = needle.as_bytes();
    if needle.is_empty() {
        return Some(0);
    }
    // Byte-wise scan: never allocates a lowercased copy of the haystack.
    // `html_to_text` calls this once per skipped block, so the old
    // whole-haystack `to_ascii_lowercase()` was O(n²) allocation on a
    // script-heavy page (up to the 5 MB web_fetch cap). The returned offset is
    // a valid `&str` index — every match starts on `<`, an ASCII char boundary.
    haystack
        .as_bytes()
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle))
}

/// Decode the handful of HTML entities that actually matter for readable text,
/// in a single left-to-right pass (one allocation, vs. one full-text copy per
/// entity in the old chained `.replace()` — the dominant cost in `html_to_text`
/// once find_ci stopped allocating). Advancing past each decoded entity makes an
/// escaped entity (`&amp;lt;`) round-trip correctly to `&lt;` without the old
/// "`&amp;` decoded last" trick — a decoded `&` is never re-scanned.
pub(super) fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    const ENTITIES: &[(&str, &str)] = &[
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", "\""),
        ("&#39;", "'"),
        ("&apos;", "'"),
        ("&nbsp;", " "),
        ("&amp;", "&"),
    ];
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let at = &rest[amp..];
        // Numeric character references (`&#39;` / `&#x27;`) — common in real pages
        // and search snippets — before the small named table.
        if let Some((decoded, len)) = decode_numeric_entity(at) {
            out.push(decoded);
            rest = &at[len..];
            continue;
        }
        match ENTITIES.iter().find(|(ent, _)| at.starts_with(ent)) {
            Some((ent, rep)) => {
                out.push_str(rep);
                rest = &at[ent.len()..];
            }
            // A bare `&` (or unknown entity): keep it and move past it.
            None => {
                out.push('&');
                rest = &at[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Decode a numeric character reference (`&#39;` decimal or `&#x27;` hex) at the
/// start of `s`, returning the char and bytes consumed (incl. the trailing `;`).
/// None if `s` isn't a well-formed numeric reference.
pub(super) fn decode_numeric_entity(s: &str) -> Option<(char, usize)> {
    let body = s.strip_prefix("&#")?;
    let (radix, digits) = match body.strip_prefix(['x', 'X']) {
        Some(rest) => (16, rest),
        None => (10, body),
    };
    let end = digits.find(';')?;
    let num = &digits[..end];
    if num.is_empty() {
        return None;
    }
    let ch = char::from_u32(u32::from_str_radix(num, radix).ok()?)?;
    // "&#" + optional "x" + digits + ";"
    let consumed = 2 + usize::from(radix == 16) + num.len() + 1;
    Some((ch, consumed))
}

/// Collapse intra-line whitespace runs and limit blank lines to one, so a tag
/// soup doesn't render as a tower of empty lines.
pub(super) fn collapse_whitespace(s: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in s.lines() {
        let trimmed = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if trimmed.is_empty() {
            if !lines.last().map(String::is_empty).unwrap_or(true) {
                lines.push(String::new());
            }
        } else {
            lines.push(trimmed);
        }
    }
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines.join("\n")
}
