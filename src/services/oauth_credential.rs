//! Shared refresh/persist orchestration for provider OAuth credentials.
//!
//! Every provider module (grok, kimi, codex, MCP) owns its wire details —
//! token endpoint, headers, error wording, PKCE vs device flow — but the
//! orchestration around them was copy-pasted per provider: refresh-if-near-
//! expiry, and write a rotated refresh token back to the key store so a later
//! process doesn't start from a revoked/stale token. These traits hold that
//! orchestration once; providers implement the wire parts.

use anyhow::Result;

use crate::services::session_store::{ApiKey, SessionStore};

/// Appended to auth-shaped OAuth failures; `aivo keys reauth` re-runs the
/// provider login in place (unlike `keys edit`, which is for plain API keys).
pub(crate) const REAUTH_HINT: &str = " — run `aivo keys reauth` to sign in again";

/// A refreshable OAuth credential. `refresh` is the provider's own token
/// exchange (endpoint, headers, expiry default all provider-specific).
pub(crate) trait OAuthCredential {
    fn is_expired(&self, skew_secs: i64) -> bool;
    async fn refresh(&mut self) -> Result<()>;
}

/// Refreshes only if near expiry; `true` if it did (caller persists).
pub(crate) async fn ensure_fresh<C: OAuthCredential>(
    creds: &mut C,
    skew_secs: i64,
) -> Result<bool> {
    if creds.is_expired(skew_secs) {
        creds.refresh().await?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// A credential stored as an aivo `ApiKey` entry whose refresh token rotates
/// on use (xAI revokes the prior one, kimi rotates without revoking) — so a
/// rotation left only in memory strands the on-disk token.
pub(crate) trait StoredOAuthCredential: OAuthCredential + Sized {
    /// Whether this `ApiKey` entry belongs to this provider.
    fn key_matches(key: &ApiKey) -> bool;
    fn from_json(json: &str) -> Result<Self>;
    fn to_json(&self) -> Result<String>;
    fn refresh_token(&self) -> &str;
}

/// Writes a rotated credential back to the key store. Best-effort; matches
/// the provider's entry by kind, then (when several) by pre-rotation refresh
/// token.
pub(crate) async fn persist_rotated_credential<C: StoredOAuthCredential>(
    store: &SessionStore,
    prev_refresh_token: &str,
    creds: &C,
) {
    let Ok(json) = creds.to_json() else {
        return;
    };
    let Ok(keys) = store.get_keys().await else {
        return;
    };
    let mut candidates: Vec<_> = keys.into_iter().filter(|k| C::key_matches(k)).collect();
    let target = match candidates.len() {
        0 => return,
        1 => candidates.pop(),
        _ => candidates.into_iter().find(|k| {
            let mut probe = k.clone();
            SessionStore::decrypt_key_secret(&mut probe).is_ok()
                && C::from_json(&probe.key)
                    .map(|c| c.refresh_token() == prev_refresh_token)
                    .unwrap_or(false)
        }),
    };
    if let Some(existing) = target {
        let _ = store
            .update_key(
                &existing.id,
                &existing.name,
                &existing.base_url,
                existing.claude_protocol,
                &json,
            )
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        expired: bool,
        refreshed: u32,
        fail: bool,
    }

    impl OAuthCredential for Fake {
        fn is_expired(&self, _skew_secs: i64) -> bool {
            self.expired
        }
        async fn refresh(&mut self) -> Result<()> {
            if self.fail {
                anyhow::bail!("refresh failed");
            }
            self.refreshed += 1;
            self.expired = false;
            Ok(())
        }
    }

    #[tokio::test]
    async fn ensure_fresh_skips_when_current() {
        let mut c = Fake {
            expired: false,
            refreshed: 0,
            fail: false,
        };
        assert!(!ensure_fresh(&mut c, 120).await.unwrap());
        assert_eq!(c.refreshed, 0);
    }

    #[tokio::test]
    async fn ensure_fresh_refreshes_when_expired_and_reports_it() {
        let mut c = Fake {
            expired: true,
            refreshed: 0,
            fail: false,
        };
        assert!(ensure_fresh(&mut c, 120).await.unwrap());
        assert_eq!(c.refreshed, 1);
        // Now fresh: a second call is a no-op.
        assert!(!ensure_fresh(&mut c, 120).await.unwrap());
        assert_eq!(c.refreshed, 1);
    }

    #[tokio::test]
    async fn ensure_fresh_propagates_refresh_errors() {
        let mut c = Fake {
            expired: true,
            refreshed: 0,
            fail: true,
        };
        assert!(ensure_fresh(&mut c, 120).await.is_err());
    }
}
