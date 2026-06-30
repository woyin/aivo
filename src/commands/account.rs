//! `aivo account` — groups login/logout and adds read-only `info`/`usage`.
//!
//! `info`/`usage` fetch plan + usage from the device-signed `/api/device/usage`
//! endpoint; identity comes from the same reconciliation `aivo info` uses.

use serde_json::{Value, json};

use crate::cli::{AccountArgs, AccountOpenArgs, AccountSubcommand};
use crate::commands::login::{AccountSync, sync_account_status};
use crate::commands::stats::{colorize_unit, format_human};
use crate::commands::{LoginCommand, LogoutCommand, trim_to_one_line};
use crate::errors::ExitCode;
use crate::services::account_store;
use crate::services::device_auth::{self, AccountUsage, UsageSummary};
use crate::services::session_store::SessionStore;
use crate::style;

/// Runs `fut` under a stderr spinner. No-op off a TTY, so `--json`/pipes stay clean.
async fn spin<F: std::future::Future>(label: &str, fut: F) -> F::Output {
    let (spinning, handle) = style::start_spinner(Some(label));
    let out = fut.await;
    style::stop_spinner(&spinning);
    let _ = handle.await;
    out
}

pub struct AccountCommand {
    session_store: SessionStore,
}

impl AccountCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(self, args: AccountArgs) -> ExitCode {
        match args.command {
            None => {
                Self::print_help(None);
                ExitCode::Success
            }
            Some(AccountSubcommand::Info(a)) => self.info(a.json).await,
            Some(AccountSubcommand::Usage(a)) => self.usage(a.json).await,
            Some(AccountSubcommand::Login(a)) => {
                LoginCommand::new(self.session_store).execute(a).await
            }
            Some(AccountSubcommand::Logout(a)) => LogoutCommand::new().execute(a).await,
            Some(AccountSubcommand::Open(a)) => Self::open(a),
        }
    }

    /// Opens (or, with `--print`, just prints) the account dashboard URL.
    fn open(args: AccountOpenArgs) -> ExitCode {
        let url = format!("{}/dashboard/", device_auth::website_base_url());
        if args.print {
            println!("{url}");
            return ExitCode::Success;
        }
        match crate::services::browser_open::open_url(&url) {
            Ok(()) => {
                println!("Opening {} in your browser.", style::blue(&url));
                ExitCode::Success
            }
            // No desktop browser (headless/SSH): print the URL so it stays usable.
            Err(_) => {
                println!("Open {} in your browser.", style::blue(&url));
                ExitCode::Success
            }
        }
    }

    /// Account identity + plan. Usage is only fetched once there's a linked account.
    async fn info(&self, as_json: bool) -> ExitCode {
        if as_json {
            // Independent device-signed calls — fetch concurrently.
            let (sync, usage) = spin(" Loading account…", async {
                tokio::join!(sync_account_status(), device_auth::fetch_account_usage())
            })
            .await;
            if let AccountUsage::Linked(s) = &usage {
                account_store::cache_plan_from(
                    s.plan.as_deref(),
                    s.is_pro,
                    s.plan_label.as_deref(),
                )
                .await;
            }
            return info_json(sync, usage);
        }

        let sync = spin(" Checking account…", sync_account_status()).await;

        match &sync {
            AccountSync::Linked(a) => println!("{} {}", style::bold("Account:"), a.display()),
            AccountSync::Unverified(Some(a)) => println!(
                "{} {} {}",
                style::bold("Account:"),
                a.display(),
                style::dim("(unverified — couldn't reach the server)")
            ),
            AccountSync::Unlinked { .. } | AccountSync::Unverified(None) => {
                println!(
                    "{} {}",
                    style::bold("Account:"),
                    style::dim("not logged in (run `aivo account login`)")
                );
                return ExitCode::Success;
            }
        }

        match spin(" Loading usage…", device_auth::fetch_account_usage()).await {
            AccountUsage::Linked(s) => {
                account_store::cache_plan_from(
                    s.plan.as_deref(),
                    s.is_pro,
                    s.plan_label.as_deref(),
                )
                .await;
                print_plan_block(&s);
            }
            AccountUsage::Unlinked | AccountUsage::Unknown => println!(
                "{} {}",
                style::bold("Plan:"),
                style::dim("(plan info unavailable)")
            ),
        }
        ExitCode::Success
    }

    /// Usage for the linked account. "not logged in" exits 0; "unreachable" exits 2.
    async fn usage(&self, as_json: bool) -> ExitCode {
        match spin(" Loading usage…", device_auth::fetch_account_usage()).await {
            AccountUsage::Unknown => {
                eprintln!(
                    "{} Couldn't reach the aivo account service.",
                    style::red("Error:")
                );
                ExitCode::NetworkError
            }
            AccountUsage::Unlinked => {
                println!(
                    "  {}",
                    style::dim("Not logged in. Run `aivo account login`.")
                );
                ExitCode::Success
            }
            AccountUsage::Linked(s) if as_json => {
                account_store::cache_plan_from(
                    s.plan.as_deref(),
                    s.is_pro,
                    s.plan_label.as_deref(),
                )
                .await;
                match serde_json::to_string_pretty(&*s) {
                    Ok(out) => {
                        println!("{out}");
                        ExitCode::Success
                    }
                    Err(e) => {
                        eprintln!("{} {e}", style::red("Error:"));
                        ExitCode::UserError
                    }
                }
            }
            AccountUsage::Linked(s) => {
                account_store::cache_plan_from(
                    s.plan.as_deref(),
                    s.is_pro,
                    s.plan_label.as_deref(),
                )
                .await;
                print_usage(&s);
                ExitCode::Success
            }
        }
    }

    pub fn print_help(sub: Option<&str>) {
        if sub == Some("login") {
            return LoginCommand::print_help();
        }
        if sub == Some("logout") {
            return LogoutCommand::print_help();
        }

        println!("{} aivo account [command]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Manage your aivo account and view your plan + usage.")
        );
        println!();
        println!("{}", style::bold("Commands:"));
        let print_cmd = |name: &str, desc: &str| {
            println!(
                "  {}  {}",
                style::cyan(format!("{:<7}", name)),
                style::dim(desc)
            );
        };
        print_cmd("info", "Identity, plan, and linked-device count");
        print_cmd(
            "usage",
            "Requests/tokens, daily caps, and per-model breakdown",
        );
        print_cmd("login", "Sign in and link this device");
        print_cmd("logout", "Sign out and unlink this device");
        print_cmd("open", "Open your account dashboard in the browser");
        println!();
        println!("{}", style::bold("Examples:"));
        for c in [
            "aivo account",
            "aivo account usage",
            "aivo account usage --json",
            "aivo account login",
        ] {
            println!("  {}", style::dim(c));
        }
    }
}

/// Plan label for display: server `plan_label`, then the `plan` slug, else `is_pro`.
fn plan_label(s: &UsageSummary) -> String {
    s.plan_label
        .as_deref()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .or_else(|| s.plan.clone())
        .unwrap_or_else(|| {
            if s.is_pro {
                "pro".into()
            } else {
                "starter".into()
            }
        })
}

/// Renders a cap as a human number, or "∞" when there's no cap (None or 0).
fn cap_or_infinity(cap: Option<u64>) -> String {
    match cap {
        Some(c) if c > 0 => format_human(c),
        _ => "∞".to_string(),
    }
}

/// Utilization percent: `0%`, `<1%` for a sub-1% sliver, else rounded.
fn pct_label(used: u64, cap: u64) -> String {
    if cap == 0 || used == 0 {
        return "0%".to_string();
    }
    let p = (used as f64 / cap as f64) * 100.0;
    if p < 1.0 {
        "<1%".to_string()
    } else {
        format!("{}%", p.round() as u64)
    }
}

/// Colours the percent by utilization: dim, yellow ≥80%, red ≥100%.
fn pct_styled(used: u64, cap: u64) -> String {
    let padded = format!("{:>4}", pct_label(used, cap));
    let ratio = if cap == 0 {
        0.0
    } else {
        used as f64 / cap as f64
    };
    if ratio >= 1.0 {
        style::red(&padded)
    } else if ratio >= 0.8 {
        style::yellow(&padded)
    } else {
        style::dim(&padded)
    }
}

/// USD like the dashboard's fmtUsd: cents under $1k, compact above.
fn fmt_usd(n: f64) -> String {
    if n >= 1_000.0 {
        format!("${}", format_human(n.round() as u64))
    } else {
        format!("${:.2}", n.max(0.0))
    }
}

fn fmt_usd_cap(cap: Option<f64>) -> String {
    match cap {
        Some(c) if c > 0.0 => fmt_usd(c),
        _ => "∞".to_string(),
    }
}

/// Whole cents, so cost reuses the integer meter/pct.
fn usd_cents(v: f64) -> u64 {
    (v * 100.0).round().max(0.0) as u64
}

/// Pre-rendered so integer and USD rows share one column-width pass.
struct MeterRow {
    name: &'static str,
    used: String,
    cap: String,
    pct: String,
    meter: String,
}

fn meter_row(
    name: &'static str,
    used: String,
    cap: String,
    used_n: u64,
    cap_n: Option<u64>,
) -> MeterRow {
    let (pct, meter) = match cap_n {
        Some(c) if c > 0 => (
            pct_styled(used_n, c),
            style::meter(used_n, c, style::METER_WIDTH),
        ),
        _ => (
            style::dim(format!("{:>4}", "—")),
            style::meter(0, 0, style::METER_WIDTH),
        ),
    };
    MeterRow {
        name,
        used,
        cap,
        pct,
        meter,
    }
}

/// Plan / subscription / device-count block for `aivo account info`.
fn print_plan_block(s: &UsageSummary) {
    println!("{} {}", style::bold("Plan:"), plan_label(s));
    if let Some(sub) = &s.subscription
        && let Some(status) = sub.get("status").and_then(Value::as_str)
    {
        let renews = sub
            .get("current_period_end")
            .and_then(Value::as_str)
            .map(|r| style::dim(format!(" · renews {r}")))
            .unwrap_or_default();
        println!("{} {}{}", style::bold("Subscription:"), status, renews);
    }
    println!("{} {}", style::bold("Linked devices:"), s.linked_devices);
}

/// RFC3339 timestamp → relative "in 5h 23m"; raw string if unparseable.
fn humanize_reset(ts: &str) -> String {
    let Ok(when) = chrono::DateTime::parse_from_rfc3339(ts) else {
        return ts.to_string();
    };
    let secs = when
        .with_timezone(&chrono::Utc)
        .signed_duration_since(chrono::Utc::now())
        .num_seconds();
    if secs <= 0 {
        return "now".to_string();
    }
    let (h, m) = (secs / 3600, (secs % 3600) / 60);
    if h > 0 {
        format!("in {h}h {m}m")
    } else if m > 0 {
        format!("in {m}m")
    } else {
        format!("in {secs}s")
    }
}

/// Full `aivo account usage` view (styled like `aivo stats`).
fn print_usage(s: &UsageSummary) {
    let mut parts = vec![plan_label(s)];
    if s.requests_total > 0 {
        parts.push(format!(
            "{} requests",
            colorize_unit(&format_human(s.requests_total))
        ));
    }
    if s.tokens_total > 0 {
        parts.push(format!(
            "{} tokens",
            colorize_unit(&format_human(s.tokens_total))
        ));
    }
    if s.searches_total > 0 {
        parts.push(format!(
            "{} searches",
            colorize_unit(&format_human(s.searches_total))
        ));
    }
    if let Some(ts) = &s.window_resets_at {
        parts.push(format!("resets {}", humanize_reset(ts)));
    }
    let header = parts.join(" · ");
    style::print_header(&header);

    println!();
    // Cost sits after Tokens, matching the dashboard.
    let rows = [
        meter_row(
            "Requests",
            format_human(s.rpd),
            cap_or_infinity(s.limits.rpd),
            s.rpd,
            s.limits.rpd,
        ),
        meter_row(
            "Tokens",
            format_human(s.tpd),
            cap_or_infinity(s.limits.tpd),
            s.tpd,
            s.limits.tpd,
        ),
        meter_row(
            "Cost",
            fmt_usd(s.cpd),
            fmt_usd_cap(s.limits.cpd),
            usd_cents(s.cpd),
            s.limits.cpd.map(usd_cents),
        ),
        meter_row(
            "Searches",
            format_human(s.searches),
            cap_or_infinity(s.limits.spd),
            s.searches,
            s.limits.spd,
        ),
        meter_row(
            "RPM",
            format_human(s.rpm),
            cap_or_infinity(s.limits.rpm),
            s.rpm,
            s.limits.rpm,
        ),
    ];
    let name_w = rows
        .iter()
        .map(|r| r.name.len())
        .max()
        .unwrap_or(0)
        .max("Today".len());
    let used_w = rows
        .iter()
        .map(|r| r.used.len())
        .max()
        .unwrap_or(0)
        .max("used".len());
    let cap_w = rows
        .iter()
        .map(|r| r.cap.len())
        .max()
        .unwrap_or(0)
        .max("limit".len());
    println!(
        "{} {} {}",
        style::bold(format!("{:<name_w$}", "Today")),
        style::dim(format!("{:>used_w$}", "used")),
        style::dim(format!("{:>cap_w$}", "limit")),
    );
    for r in &rows {
        let pn = style::cyan(format!("{:<name_w$}", r.name));
        let pu = colorize_unit(&format!("{:>used_w$}", r.used));
        let pc = colorize_unit(&format!("{:>cap_w$}", r.cap));
        println!("{pn} {pu} {pc}  {}  {}", r.pct, r.meter);
    }

    println!();
    if s.by_model.is_empty() {
        println!("{}", style::bold("By model"));
        println!("  {}", style::dim("No model usage yet."));
        return;
    }
    let mut rows: Vec<&_> = s.by_model.iter().collect();
    rows.sort_by_key(|m| std::cmp::Reverse(m.tokens));
    let max_tok = rows.iter().map(|m| m.tokens).max().unwrap_or(0);
    let names: Vec<String> = rows
        .iter()
        .map(|m| trim_to_one_line(&m.model, 28))
        .collect();
    let name_w = names
        .iter()
        .map(|n| n.chars().count())
        .max()
        .unwrap_or(0)
        .max("By model".len());
    let tok_w = rows
        .iter()
        .map(|m| format_human(m.tokens).len())
        .max()
        .unwrap_or(0)
        .max("tokens".len());
    println!(
        "{} {}",
        style::bold(format!("{:<name_w$}", "By model")),
        style::dim(format!("{:>tok_w$}", "tokens")),
    );
    let show_bar = rows.len() > 1;
    for (m, name) in rows.iter().zip(&names) {
        let pn = style::cyan(format!("{name:<name_w$}"));
        let pt = colorize_unit(&format!("{:>tok_w$}", format_human(m.tokens)));
        if show_bar {
            println!(
                "{pn} {pt}  {}",
                style::meter(m.tokens, max_tok, style::METER_WIDTH)
            );
        } else {
            println!("{pn} {pt}");
        }
    }
}

/// JSON payload for `aivo account info --json`.
fn info_json(sync: AccountSync, usage: AccountUsage) -> ExitCode {
    let account = sync.into_account().map(|a| {
        json!({
            "user_id": a.user_id,
            "email": a.email,
            "name": a.name,
            "linked_at": a.linked_at,
        })
    });
    let mut payload = json!({ "account": account });
    if let AccountUsage::Linked(s) = usage {
        payload["plan"] = json!(plan_label(&s));
        payload["is_pro"] = json!(s.is_pro);
        payload["billing_mode"] = json!(s.billing_mode);
        payload["subscription"] = s.subscription.clone().unwrap_or(Value::Null);
        payload["linked_devices"] = json!(s.linked_devices);
    }
    match serde_json::to_string_pretty(&payload) {
        Ok(out) => {
            println!("{out}");
            ExitCode::Success
        }
        Err(e) => {
            eprintln!("{} {e}", style::red("Error:"));
            ExitCode::UserError
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_or_infinity_handles_none_zero_and_value() {
        assert_eq!(cap_or_infinity(None), "∞");
        assert_eq!(cap_or_infinity(Some(0)), "∞");
        assert_eq!(cap_or_infinity(Some(30)), "30");
    }

    #[test]
    fn print_usage_zero_plan_does_not_panic() {
        print_usage(&UsageSummary::default());
    }

    #[test]
    fn fmt_usd_keeps_cents_then_compacts() {
        assert_eq!(fmt_usd(0.0), "$0.00");
        assert_eq!(fmt_usd(0.01), "$0.01");
        assert_eq!(fmt_usd(5.0), "$5.00");
        assert_eq!(fmt_usd(1234.0), "$1.2K");
    }

    #[test]
    fn fmt_usd_cap_infinity_when_uncapped() {
        assert_eq!(fmt_usd_cap(None), "∞");
        assert_eq!(fmt_usd_cap(Some(0.0)), "∞");
        assert_eq!(fmt_usd_cap(Some(5.0)), "$5.00");
    }

    #[test]
    fn usd_cents_rounds_to_whole_cents() {
        assert_eq!(usd_cents(0.0), 0);
        assert_eq!(usd_cents(0.01), 1);
        assert_eq!(usd_cents(5.0), 500);
    }

    #[test]
    fn plan_label_falls_back_to_is_pro() {
        let mut s = UsageSummary::default();
        assert_eq!(plan_label(&s), "starter");
        s.is_pro = true;
        assert_eq!(plan_label(&s), "pro");
        s.plan = Some("aivo-pro".into());
        assert_eq!(plan_label(&s), "aivo-pro");
    }

    #[test]
    fn plan_label_prefers_server_label() {
        let mut s = UsageSummary {
            plan: Some("aivo-pro".into()),
            ..Default::default()
        };
        s.plan_label = Some("Pro".into());
        assert_eq!(plan_label(&s), "Pro");
        s.plan_label = Some("  ".into());
        assert_eq!(plan_label(&s), "aivo-pro");
    }

    #[test]
    fn pct_label_boundaries() {
        assert_eq!(pct_label(0, 100), "0%");
        assert_eq!(pct_label(100, 100), "100%");
        assert_eq!(pct_label(50, 100), "50%");
        assert_eq!(pct_label(1, 1000), "<1%"); // 0.1%
        assert_eq!(pct_label(5, 1000), "<1%"); // 0.5%
        assert_eq!(pct_label(0, 0), "0%"); // no cap
    }
}
