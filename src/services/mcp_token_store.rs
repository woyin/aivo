//! Encrypted at-rest store for remote MCP servers' OAuth tokens.
//!
//! Each [`McpOAuthCredential`] is serialized to JSON, encrypted with the
//! machine-local [`session_crypto`] key (AES-256-GCM, the same scheme the chat
//! session store uses), and kept under its MCP server name in
//! `<config>/secrets/mcp_tokens.json` (mode 0600). This is deliberately separate
//! from the `keys` store so server tokens never surface as user-managed API keys
//! in `aivo keys`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::services::mcp_oauth::McpOAuthCredential;
use crate::services::session_crypto;

/// `<config>/secrets/mcp_tokens.json`.
pub fn store_path() -> Option<PathBuf> {
    Some(crate::services::paths::mcp_tokens(
        &crate::services::paths::config_dir(),
    ))
}

/// The stored credential for `server`, or `None` if absent/undecryptable. A
/// corrupt or stale entry decodes to `None` rather than erroring — the caller
/// then treats the server as needing (re-)authorization.
pub async fn load(server: &str) -> Option<McpOAuthCredential> {
    load_at(store_path().as_deref()?, server)
}

/// Save (or replace) `server`'s credential, preserving every other entry.
pub async fn save(server: &str, cred: &McpOAuthCredential) -> Result<()> {
    let path = store_path().context("no home directory for mcp_tokens.json")?;
    save_at(&path, server, cred).await
}

/// Remove `server`'s credential. `Ok(false)` if it wasn't stored.
pub async fn remove(server: &str) -> Result<bool> {
    let path = store_path().context("no home directory for mcp_tokens.json")?;
    remove_at(&path, server).await
}

// ---- path-injectable inner functions (for tests) --------------------------

/// Lenient read for the load path: a corrupt store just means tokens don't
/// load → servers show needs-auth.
fn read_root(path: &Path) -> BTreeMap<String, Value> {
    crate::services::json_store::load_or_default(path)
}

/// Strict read for the write path, so a save/remove never silently rewrites
/// the store from an empty map and destroys other servers' credentials.
fn read_root_for_write(path: &Path) -> Result<BTreeMap<String, Value>> {
    crate::services::json_store::load_for_write(path)
}

fn load_at(path: &Path, server: &str) -> Option<McpOAuthCredential> {
    let enc = read_root(path).get(server)?.as_str()?.to_string();
    let json = session_crypto::decrypt(&enc).ok()?;
    McpOAuthCredential::from_json(&json).ok()
}

async fn save_at(path: &Path, server: &str, cred: &McpOAuthCredential) -> Result<()> {
    let enc = session_crypto::encrypt(&cred.to_json()?).context("encrypt MCP credential")?;
    let mut root = read_root_for_write(path)?;
    root.insert(server.to_string(), Value::String(enc));
    write_root(path, &root).await
}

async fn remove_at(path: &Path, server: &str) -> Result<bool> {
    let mut root = read_root_for_write(path)?;
    if root.remove(server).is_none() {
        return Ok(false);
    }
    write_root(path, &root).await?;
    Ok(true)
}

async fn write_root(path: &Path, root: &BTreeMap<String, Value>) -> Result<()> {
    crate::services::json_store::save(path, root).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample(token: &str) -> McpOAuthCredential {
        McpOAuthCredential {
            access_token: token.into(),
            refresh_token: Some("rt".into()),
            token_type: "Bearer".into(),
            expiry_date: Utc::now().timestamp_millis() + 60_000,
            scope: Some("read write".into()),
            token_endpoint: "https://auth/token".into(),
            client_id: "cid".into(),
            client_secret: None,
            resource: "https://mcp/x".into(),
            authorized_url: Some("https://mcp/x".into()),
            last_refresh: Utc::now(),
        }
    }

    #[tokio::test]
    async fn save_load_remove_roundtrip_encrypted() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-tok-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("mcp_tokens.json");

        assert!(load_at(&path, "linear").is_none(), "absent → None");

        save_at(&path, "linear", &sample("tok-A")).await.unwrap();
        save_at(&path, "github", &sample("tok-B")).await.unwrap();
        assert_eq!(load_at(&path, "linear").unwrap().access_token, "tok-A");
        assert_eq!(load_at(&path, "github").unwrap().access_token, "tok-B");

        // The token must NOT be stored in clear text.
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("tok-A"), "token must be encrypted at rest");
        assert!(raw.contains("linear") && raw.contains("github"));

        // Replace one entry; the other survives.
        save_at(&path, "linear", &sample("tok-A2")).await.unwrap();
        assert_eq!(load_at(&path, "linear").unwrap().access_token, "tok-A2");
        assert_eq!(load_at(&path, "github").unwrap().access_token, "tok-B");

        assert!(remove_at(&path, "linear").await.unwrap());
        assert!(load_at(&path, "linear").is_none());
        assert!(
            !remove_at(&path, "linear").await.unwrap(),
            "second remove → false"
        );
        assert!(load_at(&path, "github").is_some(), "unrelated entry kept");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn corrupt_entry_decodes_to_none() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-tok-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mcp_tokens.json");
        std::fs::write(&path, r#"{"linear":"not-encrypted-garbage"}"#).unwrap();
        assert!(load_at(&path, "linear").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn unparseable_file_is_not_clobbered_by_save() {
        // A present-but-unparseable token file must NOT be silently rewritten
        // from an empty map (which would destroy other servers' credentials).
        let dir = std::env::temp_dir().join(format!("aivo-mcp-tok-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mcp_tokens.json");
        std::fs::write(&path, "{ this is not valid json").unwrap();

        let err = save_at(&path, "linear", &sample("tok")).await;
        assert!(err.is_err(), "save must fail closed on a corrupt store");
        assert!(remove_at(&path, "linear").await.is_err(), "remove too");
        // The original (corrupt) bytes are left untouched, not overwritten.
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{ this is not valid json"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
