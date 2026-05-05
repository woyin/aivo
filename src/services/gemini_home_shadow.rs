//! Shadow `GEMINI_CLI_HOME` for launching the native `gemini` CLI with
//! aivo's Google OAuth credentials. The shadow owns auth (we manage the
//! tokens) but transparently exposes the user's real `~/.gemini/` state
//! — chats, memory, MCP tokens, trust prompts, settings — so a launch
//! through aivo behaves like a normal launch except for the credential.
//!
//! Flow (parallel to `codex_home_shadow.rs`):
//! 1. Create a temp dir `aivo-gemini-<random>/`.
//! 2. Create `.gemini/` inside it and write the auth files we own:
//!    - `oauth_creds.json` — google-auth-library `Credentials` shape.
//!    - `google_accounts.json` — `{ active: email, old: [] }`.
//!    - `settings.json` — the user's real `settings.json` deep-merged
//!      with `security.auth.selectedType = "oauth-personal"`. The
//!      selected auth type is required: the gemini CLI opens its
//!      first-run auth picker whenever `settings.merged.security.auth
//!      .selectedType` is undefined (verified in
//!      `google-gemini/gemini-cli:packages/cli/src/core/initializer.ts`).
//!      Merging (rather than overwriting) preserves theme, telemetry,
//!      MCP config, and every other pref the user set.
//! 3. Symlink user-state files/dirs from the real `~/.gemini/` into the
//!    shadow `.gemini/`: `tmp/`, `history/`, `commands/`, `GEMINI.md`,
//!    `trustedFolders.json`, `mcp-oauth-tokens-v2.json`, `projects.json`,
//!    `state.json`, `installation_id`, `user_id`. Reads find prior
//!    state and writes persist back to the real home.
//! 4. Caller sets `GEMINI_CLI_HOME=<tempdir>` (the *parent* of `.gemini/`
//!    — the gemini CLI appends `.gemini/` itself in
//!    `Storage.getGlobalGeminiDir`) and spawns gemini.
//! 5. On exit, `read_back` reads the (possibly-rotated) `oauth_creds.json`
//!    so refreshed tokens flow back into aivo's store.
//! 6. The temp dir is removed on drop. The symlinked real-home files
//!    survive untouched.
//!
//! We deliberately do *not* set `GEMINI_FORCE_ENCRYPTED_FILE_STORAGE`: its
//! default-off value is the path that reads/writes `oauth_creds.json`
//! directly, which is exactly what we need. Setting the flag routes the
//! CLI through hybrid keychain storage (shared per-user), defeating the
//! isolation we get from a per-launch shadow dir.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::services::gemini_oauth::GeminiOAuthCredential;
use crate::services::symlink_util::{symlink_dir, symlink_file};

/// On-disk shape the gemini CLI expects at `.gemini/oauth_creds.json`.
/// Matches google-auth-library's `Credentials` exactly — all fields
/// optional so a partial write (e.g. no id_token because openid wasn't
/// requested) still parses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OAuthCredsFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_type: Option<String>,
    /// Milliseconds since epoch — google-auth-library's native format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expiry_date: Option<i64>,
}

impl OAuthCredsFile {
    pub fn from_credential(c: &GeminiOAuthCredential) -> Self {
        Self {
            access_token: Some(c.access_token.clone()),
            refresh_token: Some(c.refresh_token.clone()),
            id_token: c.id_token.clone(),
            scope: Some(c.scope.clone()),
            token_type: Some(c.token_type.clone()),
            expiry_date: Some(c.expiry_date),
        }
    }

    /// Projects the on-disk creds back to an aivo credential. Disk values
    /// win for the three token fields + scope + expiry; caller-supplied
    /// `email` and `last_refresh` are preserved because the gemini CLI
    /// doesn't track them.
    pub fn into_credential(
        self,
        email: Option<String>,
        fallback_last_refresh: chrono::DateTime<chrono::Utc>,
    ) -> GeminiOAuthCredential {
        GeminiOAuthCredential {
            access_token: self.access_token.unwrap_or_default(),
            refresh_token: self.refresh_token.unwrap_or_default(),
            id_token: self.id_token,
            scope: self.scope.unwrap_or_default(),
            token_type: self.token_type.unwrap_or_else(|| "Bearer".to_string()),
            expiry_date: self.expiry_date.unwrap_or(0),
            email,
            last_refresh: fallback_last_refresh,
        }
    }
}

/// `{ active: <email>, old: [] }` — matches the gemini CLI's
/// `UserAccountManager` schema. Pre-populated so the CLI doesn't have to
/// call `/userinfo` to discover who it's signed in as.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleAccountsFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<String>,
    #[serde(default)]
    pub old: Vec<String>,
}

/// Owns a temp dir containing `.gemini/oauth_creds.json` + friends.
/// Dropping removes the directory; callers who want to sync refreshed
/// tokens back must call `read_back` before the value is dropped.
pub struct GeminiHomeShadow {
    dir: tempfile::TempDir,
}

impl GeminiHomeShadow {
    /// Creates the temp dir and writes the three auth files. The returned
    /// `path()` is the *parent* of `.gemini/`; pass it to `GEMINI_CLI_HOME`.
    pub async fn create(creds: &GeminiOAuthCredential) -> Result<Self> {
        Self::create_with_real_home(creds, real_gemini_home()).await
    }

    async fn create_with_real_home(
        creds: &GeminiOAuthCredential,
        real_home: Option<PathBuf>,
    ) -> Result<Self> {
        let dir = tempfile::Builder::new()
            .prefix("aivo-gemini-")
            .tempdir()
            .context("create GEMINI_CLI_HOME shadow temp dir")?;

        let gemini_subdir = dir.path().join(".gemini");
        tokio::fs::create_dir_all(&gemini_subdir)
            .await
            .context("create shadow .gemini/ dir")?;

        let creds_body = serde_json::to_vec_pretty(&OAuthCredsFile::from_credential(creds))
            .context("serialize oauth_creds.json")?;
        let accounts_body = serde_json::to_vec_pretty(&GoogleAccountsFile {
            active: creds.email.clone(),
            old: Vec::new(),
        })
        .context("serialize google_accounts.json")?;
        let settings_body = serde_json::to_vec_pretty(&merged_settings(real_home.as_deref()).await)
            .context("serialize settings.json")?;

        let (creds_r, accounts_r, settings_r) = tokio::join!(
            tokio::fs::write(gemini_subdir.join("oauth_creds.json"), creds_body),
            tokio::fs::write(gemini_subdir.join("google_accounts.json"), accounts_body),
            tokio::fs::write(gemini_subdir.join("settings.json"), settings_body),
        );
        creds_r.context("write shadow oauth_creds.json")?;
        accounts_r.context("write shadow google_accounts.json")?;
        settings_r.context("write shadow settings.json")?;

        if let Some(real) = real_home {
            link_user_state(&real, &gemini_subdir).await;
        }

        Ok(Self { dir })
    }

    /// Parent of `.gemini/` — the value to set in `GEMINI_CLI_HOME`.
    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    pub fn oauth_creds_path(&self) -> PathBuf {
        self.dir.path().join(".gemini").join("oauth_creds.json")
    }

    /// Reads the on-disk `oauth_creds.json` back (after gemini exits).
    /// Missing/malformed → `Ok(None)` so callers can keep the pre-launch
    /// credential intact.
    pub async fn read_back(&self) -> Result<Option<OAuthCredsFile>> {
        let path = self.oauth_creds_path();
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(anyhow::Error::new(e).context("read shadow oauth_creds.json")),
        };
        match serde_json::from_slice::<OAuthCredsFile>(&bytes) {
            Ok(v) => Ok(Some(v)),
            Err(_) => Ok(None),
        }
    }
}

/// Resolve the user's real `.gemini/` directory, matching gemini-cli's
/// `Storage.getGlobalGeminiDir`: `$GEMINI_CLI_HOME/.gemini/` if set,
/// else `~/.gemini/`. We read the env eagerly because aivo overwrites
/// `GEMINI_CLI_HOME` to point at the shadow before spawn.
fn real_gemini_home() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("GEMINI_CLI_HOME") {
        let p = PathBuf::from(v).join(".gemini");
        if p.is_dir() {
            return Some(p);
        }
    }
    crate::services::system_env::home_dir().map(|h| h.join(".gemini"))
}

/// Loads the user's `settings.json` (if present) and deep-merges
/// `security.auth.selectedType = "oauth-personal"` into it. The selected
/// auth type is required: gemini-cli opens its first-run auth picker
/// whenever `settings.merged.security.auth.selectedType` is undefined,
/// regardless of `oauth_creds.json` being populated. Merging (rather
/// than overwriting) preserves theme, telemetry, MCP, and every other
/// pref the user set in their real CLI.
async fn merged_settings(real_home: Option<&Path>) -> serde_json::Value {
    let mut base = match real_home {
        Some(home) => match tokio::fs::read(home.join("settings.json")).await {
            Ok(bytes) => serde_json::from_slice::<serde_json::Value>(&bytes)
                .unwrap_or_else(|_| serde_json::json!({})),
            Err(_) => serde_json::json!({}),
        },
        None => serde_json::json!({}),
    };
    if !base.is_object() {
        base = serde_json::json!({});
    }
    let security = base
        .as_object_mut()
        .unwrap()
        .entry("security")
        .or_insert_with(|| serde_json::json!({}));
    if !security.is_object() {
        *security = serde_json::json!({});
    }
    let auth = security
        .as_object_mut()
        .unwrap()
        .entry("auth")
        .or_insert_with(|| serde_json::json!({}));
    if !auth.is_object() {
        *auth = serde_json::json!({});
    }
    auth.as_object_mut()
        .unwrap()
        .insert("selectedType".into(), serde_json::json!("oauth-personal"));
    base
}

/// Best-effort: link the user-state files of `~/.gemini/` into the
/// shadow `.gemini/` so chat resume, memory, trust state, MCP tokens,
/// and project bookkeeping all work as if the user launched gemini
/// directly — and writes during this session persist back to the real
/// home. Each link is independent; failures are silent.
async fn link_user_state(real_home: &Path, shadow_gemini: &Path) {
    for dir in ["tmp", "history", "commands"] {
        let real = real_home.join(dir);
        let _ = tokio::fs::create_dir_all(&real).await;
        if real.is_dir() {
            let _ = symlink_dir(&real, &shadow_gemini.join(dir)).await;
        }
    }

    for file in [
        "GEMINI.md",
        "trustedFolders.json",
        "mcp-oauth-tokens-v2.json",
        "projects.json",
        "state.json",
        "installation_id",
        "user_id",
    ] {
        let real = real_home.join(file);
        if real.exists() {
            let _ = symlink_file(&real, &shadow_gemini.join(file)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample_cred() -> GeminiOAuthCredential {
        GeminiOAuthCredential {
            access_token: "at".into(),
            refresh_token: "rt".into(),
            id_token: None,
            scope: "s".into(),
            token_type: "Bearer".into(),
            expiry_date: Utc::now().timestamp_millis() + 3_600_000,
            email: Some("a@b.com".into()),
            last_refresh: Utc::now(),
        }
    }

    async fn isolated_shadow(c: &GeminiOAuthCredential) -> GeminiHomeShadow {
        GeminiHomeShadow::create_with_real_home(c, None)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn creates_dot_gemini_subdir_and_files() {
        let c = sample_cred();
        let shadow = isolated_shadow(&c).await;
        let dot = shadow.path().join(".gemini");
        assert!(dot.is_dir());
        assert!(dot.join("oauth_creds.json").is_file());
        assert!(dot.join("google_accounts.json").is_file());
        assert!(dot.join("settings.json").is_file());
    }

    #[tokio::test]
    async fn settings_selects_oauth_personal_auth_type() {
        // The gemini CLI opens its auth picker whenever
        // settings.security.auth.selectedType is undefined. Regression guard.
        let c = sample_cred();
        let shadow = isolated_shadow(&c).await;
        let body = tokio::fs::read_to_string(shadow.path().join(".gemini/settings.json"))
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            parsed["security"]["auth"]["selectedType"].as_str(),
            Some("oauth-personal")
        );
    }

    #[tokio::test]
    async fn roundtrip_preserves_tokens() {
        let c = sample_cred();
        let shadow = isolated_shadow(&c).await;
        let back = shadow.read_back().await.unwrap().unwrap();
        assert_eq!(back.access_token.as_deref(), Some("at"));
        assert_eq!(back.refresh_token.as_deref(), Some("rt"));
        assert_eq!(back.scope.as_deref(), Some("s"));
        assert_eq!(back.token_type.as_deref(), Some("Bearer"));
        assert_eq!(back.expiry_date, Some(c.expiry_date));
    }

    #[tokio::test]
    async fn google_accounts_file_has_active_email() {
        let c = sample_cred();
        let shadow = isolated_shadow(&c).await;
        let body = tokio::fs::read_to_string(shadow.path().join(".gemini/google_accounts.json"))
            .await
            .unwrap();
        let parsed: GoogleAccountsFile = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.active.as_deref(), Some("a@b.com"));
        assert!(parsed.old.is_empty());
    }

    #[tokio::test]
    async fn read_back_handles_missing_file() {
        let c = sample_cred();
        let shadow = isolated_shadow(&c).await;
        tokio::fs::remove_file(shadow.oauth_creds_path())
            .await
            .unwrap();
        assert!(shadow.read_back().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn read_back_handles_malformed_json() {
        let c = sample_cred();
        let shadow = isolated_shadow(&c).await;
        tokio::fs::write(shadow.oauth_creds_path(), b"{not json")
            .await
            .unwrap();
        assert!(shadow.read_back().await.unwrap().is_none());
    }

    #[test]
    fn into_credential_prefers_disk_tokens() {
        let c = sample_cred();
        let disk = OAuthCredsFile {
            access_token: Some("new-at".into()),
            refresh_token: Some("new-rt".into()),
            id_token: None,
            scope: Some("new-s".into()),
            token_type: Some("Bearer".into()),
            expiry_date: Some(123),
        };
        let back = disk.into_credential(c.email.clone(), c.last_refresh);
        assert_eq!(back.access_token, "new-at");
        assert_eq!(back.refresh_token, "new-rt");
        assert_eq!(back.scope, "new-s");
        assert_eq!(back.expiry_date, 123);
        assert_eq!(back.email, c.email);
        assert_eq!(back.last_refresh, c.last_refresh);
    }

    #[tokio::test]
    async fn temp_dir_is_removed_on_drop() {
        let c = sample_cred();
        let path = {
            let shadow = isolated_shadow(&c).await;
            shadow.path().to_path_buf()
        };
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn settings_merge_preserves_user_prefs() {
        // Real settings.json has unrelated user prefs and a *different*
        // selectedType — merged shadow output must keep the prefs and
        // overwrite only selectedType.
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        let real_gemini = real.path().join(".gemini");
        tokio::fs::create_dir_all(&real_gemini).await.unwrap();
        let user_settings = serde_json::json!({
            "ui": { "theme": "Dracula" },
            "telemetry": { "enabled": false },
            "security": { "auth": { "selectedType": "stale" } },
            "mcpServers": { "foo": { "command": "x" } }
        });
        tokio::fs::write(
            real_gemini.join("settings.json"),
            serde_json::to_vec(&user_settings).unwrap(),
        )
        .await
        .unwrap();

        let shadow = GeminiHomeShadow::create_with_real_home(&c, Some(real_gemini.clone()))
            .await
            .unwrap();

        let body = tokio::fs::read(shadow.path().join(".gemini/settings.json"))
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["ui"]["theme"].as_str(), Some("Dracula"));
        assert_eq!(parsed["telemetry"]["enabled"], false);
        assert_eq!(parsed["mcpServers"]["foo"]["command"].as_str(), Some("x"));
        assert_eq!(
            parsed["security"]["auth"]["selectedType"].as_str(),
            Some("oauth-personal")
        );
    }

    #[tokio::test]
    async fn shadow_exposes_chats_and_memory() {
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        let real_gemini = real.path().join(".gemini");
        let chats_dir = real_gemini.join("tmp/projhash/chats");
        tokio::fs::create_dir_all(&chats_dir).await.unwrap();
        tokio::fs::write(chats_dir.join("session-old.json"), b"{}")
            .await
            .unwrap();
        tokio::fs::write(real_gemini.join("GEMINI.md"), b"my context")
            .await
            .unwrap();
        tokio::fs::write(real_gemini.join("trustedFolders.json"), b"{}")
            .await
            .unwrap();

        let shadow = GeminiHomeShadow::create_with_real_home(&c, Some(real_gemini.clone()))
            .await
            .unwrap();
        let dot = shadow.path().join(".gemini");

        let chat = tokio::fs::read(dot.join("tmp/projhash/chats/session-old.json"))
            .await
            .unwrap();
        assert_eq!(chat, b"{}");
        let mem = tokio::fs::read(dot.join("GEMINI.md")).await.unwrap();
        assert_eq!(mem, b"my context");
        assert!(dot.join("trustedFolders.json").exists());
    }

    #[tokio::test]
    async fn new_chat_session_persists_back_to_real_home() {
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        let real_gemini = real.path().join(".gemini");
        tokio::fs::create_dir_all(&real_gemini).await.unwrap();

        let shadow = GeminiHomeShadow::create_with_real_home(&c, Some(real_gemini.clone()))
            .await
            .unwrap();

        let new_chat = shadow
            .path()
            .join(".gemini/tmp/projhash/chats/session-new.json");
        tokio::fs::create_dir_all(new_chat.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&new_chat, b"new").await.unwrap();

        let in_real = tokio::fs::read(real_gemini.join("tmp/projhash/chats/session-new.json"))
            .await
            .unwrap();
        assert_eq!(in_real, b"new");
    }

    #[tokio::test]
    async fn missing_user_settings_still_emits_oauth_personal() {
        // Real home exists but has no settings.json → shadow falls back
        // to a minimal {security:{auth:{selectedType}}} object so the
        // first-run picker still doesn't trigger.
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        let real_gemini = real.path().join(".gemini");
        tokio::fs::create_dir_all(&real_gemini).await.unwrap();

        let shadow = GeminiHomeShadow::create_with_real_home(&c, Some(real_gemini))
            .await
            .unwrap();
        let body = tokio::fs::read(shadow.path().join(".gemini/settings.json"))
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            parsed["security"]["auth"]["selectedType"].as_str(),
            Some("oauth-personal")
        );
    }
}
