//! Local record of the signed-in getaivo.dev account.
//!
//! Written by `aivo login` after the device is linked; read by `aivo info`.
//! This is display metadata only — there is no secret here (the Ed25519 device
//! key remains the credential), so it lives as a plain JSON file at
//! `<config>/secrets/account.json` (mode 0600), separate from the encrypted
//! `config.json` and untouched by other aivo commands.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// The account this device is linked to. `email`/`name` are best-effort —
/// the server may omit them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Account {
    pub user_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub linked_at: String,
    /// Canonical plan slug (`aivo-pro`, …), or `None` for the free starter tier.
    /// Cached from the last usage fetch; absent in pre-plan `account.json` files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<String>,
    /// Human plan label from the gateway; display prefers it over `plan`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_label: Option<String>,
}

impl Account {
    /// Best label for the account: email, then name, then the opaque user id.
    pub fn display(&self) -> &str {
        self.email
            .as_deref()
            .or(self.name.as_deref())
            .unwrap_or(&self.user_id)
    }
}

fn account_path() -> Option<PathBuf> {
    Some(crate::services::paths::account_json(
        &crate::services::paths::config_dir(),
    ))
}

/// Loads the stored account, or `None` if not logged in / unreadable.
pub fn load() -> Option<Account> {
    load_from(&account_path()?)
}

fn load_from(path: &Path) -> Option<Account> {
    crate::services::json_store::load_optional(path)
}

/// Persists the account record atomically (0600).
pub async fn save(account: &Account) -> Result<()> {
    let path = account_path().ok_or_else(|| anyhow::anyhow!("could not resolve home directory"))?;
    crate::services::json_store::save(&path, account).await
}

/// Removes the stored account. Returns true if a record was present.
pub fn clear() -> bool {
    match account_path() {
        Some(path) => std::fs::remove_file(path).is_ok(),
        None => false,
    }
}

/// Canonical plan slug for display: `Some("aivo-pro")`, `Some("aivo-friend")`, …
/// or `None` for the free starter tier (no plan, explicit `starter`, or empty).
/// Normalizes an optional `aivo-` prefix and lowercases the tier so server
/// variants (`"pro"`, `"aivo-Pro"`, `"aivo-pro"`) all collapse to `aivo-pro`.
pub fn canonical_plan(plan: Option<&str>, is_pro: bool) -> Option<String> {
    match plan.map(str::trim).filter(|s| !s.is_empty()) {
        Some(p) => {
            let bare = p.strip_prefix("aivo-").unwrap_or(p);
            if bare.eq_ignore_ascii_case("starter") {
                None
            } else {
                Some(format!("aivo-{}", bare.to_ascii_lowercase()))
            }
        }
        None if is_pro => Some("aivo-pro".to_string()),
        None => None,
    }
}

/// Updates the cached `plan` + human `plan_label` on the stored account
/// (load-modify-save). No-op when not logged in (no `account.json`) or current.
pub async fn update_plan(plan: Option<String>, plan_label: Option<String>) -> Result<()> {
    let Some(mut account) = load() else {
        return Ok(());
    };
    if account.plan == plan && account.plan_label == plan_label {
        return Ok(());
    }
    account.plan = plan;
    account.plan_label = plan_label;
    save(&account).await
}

/// Canonical `(plan, plan_label)` from a usage fetch; the label is kept only on
/// a paid plan (blank → dropped, so display falls back to the slug).
pub fn canonical_plan_with_label(
    plan: Option<&str>,
    is_pro: bool,
    plan_label: Option<&str>,
) -> (Option<String>, Option<String>) {
    let canonical = canonical_plan(plan, is_pro);
    let label = canonical.as_ref().and_then(|_| {
        plan_label
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    });
    (canonical, label)
}

/// Best-effort refresh of the cached plan + label from a usage fetch.
pub async fn cache_plan_from(plan: Option<&str>, is_pro: bool, plan_label: Option<&str>) {
    let (canonical, label) = canonical_plan_with_label(plan, is_pro, plan_label);
    let _ = update_plan(canonical, label).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample() -> Account {
        Account {
            user_id: "u1".into(),
            email: Some("a@b.co".into()),
            name: Some("Ann".into()),
            linked_at: "2026-06-25T00:00:00Z".into(),
            plan: None,
            plan_label: None,
        }
    }

    #[test]
    fn round_trips_through_json() {
        let a = sample();
        let json = serde_json::to_vec_pretty(&a).unwrap();
        let back: Account = serde_json::from_slice(&json).unwrap();
        assert_eq!(a, back);
    }

    #[test]
    fn load_from_missing_is_none() {
        let dir = TempDir::new().unwrap();
        assert!(load_from(&dir.path().join("account.json")).is_none());
    }

    #[test]
    fn load_from_reads_written_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("account.json");
        std::fs::write(&path, serde_json::to_vec(&sample()).unwrap()).unwrap();
        assert_eq!(load_from(&path), Some(sample()));
    }

    #[test]
    fn email_and_name_are_optional() {
        let json = br#"{"user_id":"u","linked_at":"t"}"#;
        let a: Account = serde_json::from_slice(json).unwrap();
        assert_eq!(a.user_id, "u");
        assert!(a.email.is_none());
        assert!(a.name.is_none());
        assert!(a.plan.is_none());
        assert_eq!(a.display(), "u");
    }

    #[test]
    fn round_trips_with_plan() {
        let mut a = sample();
        a.plan = Some("aivo-pro".into());
        a.plan_label = Some("Pro".into());
        let json = serde_json::to_vec_pretty(&a).unwrap();
        let back: Account = serde_json::from_slice(&json).unwrap();
        assert_eq!(a, back);
        assert_eq!(back.plan.as_deref(), Some("aivo-pro"));
    }

    #[test]
    fn canonical_plan_starter_and_empty_are_none() {
        assert_eq!(canonical_plan(None, false), None);
        assert_eq!(canonical_plan(Some(""), false), None);
        assert_eq!(canonical_plan(Some("  "), false), None);
        assert_eq!(canonical_plan(Some("starter"), false), None);
        assert_eq!(canonical_plan(Some("aivo-starter"), false), None);
    }

    #[test]
    fn canonical_plan_normalizes_paid_tiers() {
        assert_eq!(
            canonical_plan(Some("aivo-pro"), false).as_deref(),
            Some("aivo-pro")
        );
        assert_eq!(
            canonical_plan(Some("pro"), false).as_deref(),
            Some("aivo-pro")
        );
        assert_eq!(
            canonical_plan(Some("aivo-Pro"), false).as_deref(),
            Some("aivo-pro")
        );
        assert_eq!(
            canonical_plan(Some("aivo-friend"), false).as_deref(),
            Some("aivo-friend")
        );
    }

    #[test]
    fn canonical_plan_is_pro_fallback_when_unnamed() {
        assert_eq!(canonical_plan(None, true).as_deref(), Some("aivo-pro"));
        // An explicit plan name wins over the boolean.
        assert_eq!(
            canonical_plan(Some("aivo-friend"), true).as_deref(),
            Some("aivo-friend")
        );
        // Explicit starter stays starter even if the flag disagrees.
        assert_eq!(canonical_plan(Some("starter"), true), None);
    }

    #[test]
    fn display_prefers_email_then_name() {
        assert_eq!(sample().display(), "a@b.co");
        let mut a = sample();
        a.email = None;
        assert_eq!(a.display(), "Ann");
    }
}
