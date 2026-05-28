//! Shadow `CODEX_HOME` for launching native Codex surfaces with aivo-managed
//! credentials. The shadow owns auth/config overlays (we manage the tokens and
//! provider flags) but transparently exposes the user's real `~/.codex/` state
//! — sessions, history, AGENTS.md, memories — so a launch through aivo behaves
//! like a normal launch except for credential/config isolation.
//!
//! Flow (mirrors the Pi `PI_CODING_AGENT_DIR` pattern in
//! `launch_runtime.rs::write_pi_agent_dir`):
//! 1. Create a temp dir `aivo-codex-<random>/` for CLI or a persistent
//!    aivo-owned dir for Codex Desktop App.
//! 2. If using Codex OAuth, write `auth.json` in the native codex `AuthDotJson` schema
//!    (see `openai/codex: codex-rs/login/src/token_data.rs`).
//! 3. Copy the user's `config.toml` into the shadow. Persistent app homes are
//!    seeded only once so app-side settings stay in the aivo-owned copy.
//! 4. Symlink user-state files/dirs from the real `~/.codex/`:
//!    `sessions/`, `memories/`, `skills/`, `plugins/`, `rules/`,
//!    `prompts/`, `history.jsonl`, `AGENTS.md`, `installation_id`,
//!    `models_cache.json`, `.codex-global-state.json`. Reads find prior
//!    state and writes persist back to the real home.
//! 5. Caller sets `CODEX_HOME=<dir>` on the child env and spawns codex.
//! 6. On exit, `read_back` reads the (possibly-rotated) auth.json so the
//!    refreshed tokens can be persisted back into aivo's store.
//! 7. The temp dir is removed for CLI launches. Persistent app homes remain so
//!    a detached GUI can keep using the same isolated CODEX_HOME.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::services::codex_oauth::CodexOAuthCredential;
use crate::services::symlink_util::{symlink_dir, symlink_file};

/// On-disk shape expected by the native `codex` CLI. Keep the JSON stable
/// across codex versions: extra fields are preserved on read (via
/// `serde_json::Value`) so round-trip doesn't clobber future additions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthDotJson {
    #[serde(rename = "OPENAI_API_KEY", default)]
    pub openai_api_key: Option<String>,
    pub tokens: TokenData,
    #[serde(default)]
    pub last_refresh: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenData {
    pub id_token: String,
    pub access_token: String,
    pub refresh_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
}

impl AuthDotJson {
    pub fn from_credential(c: &CodexOAuthCredential) -> Self {
        Self {
            openai_api_key: None,
            tokens: TokenData {
                id_token: c.id_token.clone(),
                access_token: c.access_token.clone(),
                refresh_token: c.refresh_token.clone(),
                account_id: c.account_id.clone(),
            },
            last_refresh: Some(c.last_refresh),
        }
    }

    /// Projects the on-disk auth.json back to an aivo credential, preferring
    /// the disk values for the three tokens and `account_id`, and preserving
    /// the passed-in `email` + `expires_at` (codex doesn't track either
    /// separately).
    pub fn into_credential(
        self,
        email: Option<String>,
        fallback_expires_at: DateTime<Utc>,
    ) -> CodexOAuthCredential {
        let last_refresh = self.last_refresh.unwrap_or_else(Utc::now);
        CodexOAuthCredential {
            id_token: self.tokens.id_token,
            access_token: self.tokens.access_token,
            refresh_token: self.tokens.refresh_token,
            account_id: self.tokens.account_id,
            email,
            // codex doesn't persist `expires_at`; aivo will refresh-on-demand
            // before next launch, so a stale value here is fine.
            expires_at: fallback_expires_at,
            last_refresh,
        }
    }
}

/// Owns a shadow CODEX_HOME. CLI launches use a temp dir that is removed on
/// drop; desktop-app launches use a persistent aivo-owned dir because the GUI
/// can outlive the `codex app` launcher process.
pub struct CodexHomeShadow {
    dir: ShadowDir,
}

enum ShadowDir {
    Temp(tempfile::TempDir),
    Persistent(PathBuf),
}

#[derive(Clone, Copy)]
enum ConfigSeedMode {
    Always,
    IfMissing,
}

impl CodexHomeShadow {
    /// Creates the temp dir and writes `auth.json`.
    /// Also copies the user's `config.toml` and links `sessions/` +
    /// `history.jsonl` from the real `~/.codex/` so settings, the
    /// `/resume` picker, and ↑-arrow input recall all work — and any new
    /// rollouts written during this launch persist back to the real home.
    pub async fn create(creds: &CodexOAuthCredential) -> Result<Self> {
        Self::create_with_real_home(creds, real_codex_home()).await
    }

    async fn create_with_real_home(
        creds: &CodexOAuthCredential,
        real_home: Option<PathBuf>,
    ) -> Result<Self> {
        let dir = tempfile::Builder::new()
            .prefix("aivo-codex-")
            .tempdir()
            .context("create CODEX_HOME shadow temp dir")?;

        initialize_shadow_home(
            dir.path(),
            Some(creds),
            real_home.as_deref(),
            ConfigSeedMode::Always,
        )
        .await?;

        Ok(Self {
            dir: ShadowDir::Temp(dir),
        })
    }

    pub async fn create_persistent(
        creds: &CodexOAuthCredential,
        config_dir: &Path,
        key_id: &str,
    ) -> Result<Self> {
        Self::create_persistent_with_real_home(Some(creds), config_dir, key_id, real_codex_home())
            .await
    }

    pub async fn create_persistent_without_auth(config_dir: &Path, key_id: &str) -> Result<Self> {
        Self::create_persistent_with_real_home(None, config_dir, key_id, real_codex_home()).await
    }

    async fn create_persistent_with_real_home(
        creds: Option<&CodexOAuthCredential>,
        config_dir: &Path,
        key_id: &str,
        real_home: Option<PathBuf>,
    ) -> Result<Self> {
        let dir = Self::persistent_path(config_dir, key_id);
        tokio::fs::create_dir_all(&dir)
            .await
            .with_context(|| format!("create persistent CODEX_HOME shadow {}", dir.display()))?;
        initialize_shadow_home(&dir, creds, real_home.as_deref(), ConfigSeedMode::IfMissing)
            .await?;
        Ok(Self {
            dir: ShadowDir::Persistent(dir),
        })
    }

    pub fn persistent_path(config_dir: &Path, key_id: &str) -> PathBuf {
        config_dir
            .join("codex-app-home")
            .join(safe_codex_app_home_key(key_id))
    }

    pub fn path(&self) -> &Path {
        match &self.dir {
            ShadowDir::Temp(dir) => dir.path(),
            ShadowDir::Persistent(path) => path.as_path(),
        }
    }

    pub fn auth_path(&self) -> PathBuf {
        self.path().join("auth.json")
    }

    /// Reads the on-disk auth.json back (after codex exits). If the file is
    /// missing or malformed — codex crashed, user killed it, etc. —
    /// returns `Ok(None)` so the caller can keep the pre-launch credential
    /// intact.
    pub async fn read_back(&self) -> Result<Option<AuthDotJson>> {
        Self::read_auth_path(self.auth_path()).await
    }

    pub async fn read_auth_path(path: impl AsRef<Path>) -> Result<Option<AuthDotJson>> {
        let path = path.as_ref();
        let bytes = match tokio::fs::read(path).await {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(anyhow::Error::new(e).context("read shadow auth.json")),
        };
        match serde_json::from_slice::<AuthDotJson>(&bytes) {
            Ok(v) => Ok(Some(v)),
            Err(_) => Ok(None),
        }
    }
}

async fn initialize_shadow_home(
    shadow: &Path,
    creds: Option<&CodexOAuthCredential>,
    real_home: Option<&Path>,
    config_seed_mode: ConfigSeedMode,
) -> Result<()> {
    if let Some(creds) = creds {
        let auth = AuthDotJson::from_credential(creds);
        let body = serde_json::to_vec_pretty(&auth).context("serialize auth.json")?;
        tokio::fs::write(shadow.join("auth.json"), body)
            .await
            .context("write shadow auth.json")?;
    } else {
        match tokio::fs::remove_file(shadow.join("auth.json")).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => {}
        }
    }

    if let Some(real_home) = real_home {
        seed_config_toml(real_home, shadow, config_seed_mode).await;
        link_session_state(real_home, shadow).await;
    }

    Ok(())
}

async fn seed_config_toml(real_home: &Path, shadow: &Path, mode: ConfigSeedMode) {
    let src = real_home.join("config.toml");
    if !src.exists() {
        return;
    }
    let dest = shadow.join("config.toml");
    if matches!(mode, ConfigSeedMode::IfMissing) && dest.exists() {
        return;
    }
    let _ = tokio::fs::copy(&src, dest).await;
}

fn safe_codex_app_home_key(key_id: &str) -> String {
    let mut out = String::with_capacity(key_id.len());
    for ch in key_id.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "default".to_string()
    } else {
        out
    }
}

fn real_codex_home() -> Option<std::path::PathBuf> {
    if let Ok(v) = std::env::var("CODEX_HOME") {
        let p = std::path::PathBuf::from(v);
        if p.is_dir() {
            return Some(p);
        }
    }
    crate::services::system_env::home_dir().map(|h| h.join(".codex"))
}

/// Best-effort: link the user-state pieces of `~/.codex/` into the shadow
/// so codex sees prior `/resume` rollouts, ↑-arrow input history, the
/// user-level `AGENTS.md`, the `/memory` store, installed skills/plugins,
/// execpolicy rules, and slash-command prompts — and any new entries
/// written during this launch persist back to the real home. Each link is
/// independent; failures are silent so codex falls back to a fresh shadow
/// location for the missing piece.
///
/// We deliberately don't link the codex state DBs (`state_*.sqlite`,
/// `logs_*.sqlite`); codex rebuilds them by scanning the symlinked
/// `sessions/`, and skipping them avoids cross-process SQLite WAL/SHM
/// hazards.
async fn link_session_state(real_home: &Path, shadow: &Path) {
    for dir in [
        "sessions", "memories", "skills", "plugins", "rules", "prompts",
    ] {
        let real = real_home.join(dir);
        let _ = tokio::fs::create_dir_all(&real).await;
        if real.is_dir() {
            let dest = shadow.join(dir);
            replace_with_symlink_dir(&real, &dest).await;
        }
    }

    for file in [
        "history.jsonl",
        "AGENTS.md",
        "installation_id",
        "models_cache.json",
        ".codex-global-state.json",
    ] {
        let real = real_home.join(file);
        if real.exists() {
            let dest = shadow.join(file);
            replace_with_symlink_file(&real, &dest).await;
        }
    }
}

/// Re-establishes a `dest -> real` file symlink, clearing any stale regular
/// file left behind by a prior CodexApp launch (e.g. `models_cache.json` after
/// `install_codex_app_models_cache` overwrote the symlink with a per-key
/// cache). Without this step, the persistent shadow keeps the stale file on
/// the next launch even if discovery fails to write a fresh one.
async fn replace_with_symlink_file(real: &Path, dest: &Path) {
    if let Ok(meta) = tokio::fs::symlink_metadata(dest).await {
        if meta.file_type().is_symlink() {
            // Already a symlink — leave it; the OS will resolve it to `real`.
            return;
        }
        // Regular file (or dir) sitting at the target path. Unlink so the
        // symlink_file call below has a clean slot. Failure is fine — likely
        // a permission/race we can't usefully recover from.
        let _ = tokio::fs::remove_file(dest).await;
    }
    let _ = symlink_file(real, dest).await;
}

async fn replace_with_symlink_dir(real: &Path, dest: &Path) {
    if let Ok(meta) = tokio::fs::symlink_metadata(dest).await {
        if meta.file_type().is_symlink() {
            return;
        }
        let _ = tokio::fs::remove_dir_all(dest).await;
    }
    let _ = symlink_dir(real, dest).await;
}

/// Returns true if the on-disk tokens differ from `original` in any field
/// codex may have rotated.
pub fn tokens_changed(original: &CodexOAuthCredential, disk: &AuthDotJson) -> bool {
    original.refresh_token != disk.tokens.refresh_token
        || original.access_token != disk.tokens.access_token
        || original.id_token != disk.tokens.id_token
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    fn sample_cred() -> CodexOAuthCredential {
        CodexOAuthCredential {
            id_token: "id".into(),
            access_token: "at".into(),
            refresh_token: "rt".into(),
            account_id: Some("acct_1".into()),
            email: Some("a@b.com".into()),
            expires_at: Utc::now() + ChronoDuration::seconds(3600),
            last_refresh: Utc::now(),
        }
    }

    async fn isolated_shadow(c: &CodexOAuthCredential) -> CodexHomeShadow {
        CodexHomeShadow::create_with_real_home(c, None)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn roundtrip_preserves_tokens() {
        let c = sample_cred();
        let shadow = isolated_shadow(&c).await;
        let back = shadow.read_back().await.unwrap().unwrap();
        assert_eq!(back.tokens.id_token, c.id_token);
        assert_eq!(back.tokens.access_token, c.access_token);
        assert_eq!(back.tokens.refresh_token, c.refresh_token);
        assert_eq!(back.tokens.account_id, c.account_id);
        assert!(back.openai_api_key.is_none());
    }

    #[tokio::test]
    async fn read_back_handles_missing_file() {
        let c = sample_cred();
        let shadow = isolated_shadow(&c).await;
        tokio::fs::remove_file(shadow.auth_path()).await.unwrap();
        assert!(shadow.read_back().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn read_back_handles_malformed_json() {
        let c = sample_cred();
        let shadow = isolated_shadow(&c).await;
        tokio::fs::write(shadow.auth_path(), b"{not json")
            .await
            .unwrap();
        assert!(shadow.read_back().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn detects_rotated_tokens() {
        let c = sample_cred();
        let shadow = isolated_shadow(&c).await;
        let mut disk = shadow.read_back().await.unwrap().unwrap();
        assert!(!tokens_changed(&c, &disk));
        disk.tokens.refresh_token = "rotated".into();
        assert!(tokens_changed(&c, &disk));
    }

    #[test]
    fn into_credential_preserves_metadata() {
        let c = sample_cred();
        let mut auth = AuthDotJson::from_credential(&c);
        auth.tokens.access_token = "new-at".into();
        let back = auth.into_credential(c.email.clone(), c.expires_at);
        assert_eq!(back.access_token, "new-at");
        assert_eq!(back.email, c.email);
        assert_eq!(back.expires_at, c.expires_at);
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
    async fn shadow_sees_prior_sessions_via_link() {
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        let prior = real.path().join("sessions/2026/01/rollout-abc.jsonl");
        tokio::fs::create_dir_all(prior.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&prior, b"prior").await.unwrap();

        let shadow = CodexHomeShadow::create_with_real_home(&c, Some(real.path().to_path_buf()))
            .await
            .unwrap();

        let via_shadow = shadow.path().join("sessions/2026/01/rollout-abc.jsonl");
        let bytes = tokio::fs::read(&via_shadow).await.unwrap();
        assert_eq!(bytes, b"prior");
    }

    #[tokio::test]
    async fn new_session_writes_persist_to_real_home() {
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();

        let shadow = CodexHomeShadow::create_with_real_home(&c, Some(real.path().to_path_buf()))
            .await
            .unwrap();

        let new_rollout = shadow.path().join("sessions/2026/05/rollout-new.jsonl");
        tokio::fs::create_dir_all(new_rollout.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&new_rollout, b"new").await.unwrap();

        let real_path = real.path().join("sessions/2026/05/rollout-new.jsonl");
        let bytes = tokio::fs::read(&real_path).await.unwrap();
        assert_eq!(bytes, b"new");
    }

    #[tokio::test]
    async fn history_jsonl_appends_persist_to_real_home() {
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        let real_history = real.path().join("history.jsonl");
        tokio::fs::write(&real_history, b"old\n").await.unwrap();

        let shadow = CodexHomeShadow::create_with_real_home(&c, Some(real.path().to_path_buf()))
            .await
            .unwrap();

        let shadow_history = shadow.path().join("history.jsonl");
        let mut existing = tokio::fs::read(&shadow_history).await.unwrap();
        assert_eq!(existing, b"old\n");
        existing.extend_from_slice(b"new\n");
        tokio::fs::write(&shadow_history, existing).await.unwrap();

        let bytes = tokio::fs::read(&real_history).await.unwrap();
        assert_eq!(bytes, b"old\nnew\n");
    }

    #[tokio::test]
    async fn shadow_exposes_user_agents_md() {
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        tokio::fs::write(real.path().join("AGENTS.md"), b"be excellent")
            .await
            .unwrap();

        let shadow = CodexHomeShadow::create_with_real_home(&c, Some(real.path().to_path_buf()))
            .await
            .unwrap();

        let bytes = tokio::fs::read(shadow.path().join("AGENTS.md"))
            .await
            .unwrap();
        assert_eq!(bytes, b"be excellent");
    }

    #[tokio::test]
    async fn memories_link_persists_writes_back() {
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        tokio::fs::create_dir_all(real.path().join("memories"))
            .await
            .unwrap();
        tokio::fs::write(real.path().join("memories/old.md"), b"old")
            .await
            .unwrap();

        let shadow = CodexHomeShadow::create_with_real_home(&c, Some(real.path().to_path_buf()))
            .await
            .unwrap();

        let via_shadow = tokio::fs::read(shadow.path().join("memories/old.md"))
            .await
            .unwrap();
        assert_eq!(via_shadow, b"old");

        tokio::fs::write(shadow.path().join("memories/new.md"), b"new")
            .await
            .unwrap();
        let in_real = tokio::fs::read(real.path().join("memories/new.md"))
            .await
            .unwrap();
        assert_eq!(in_real, b"new");
    }

    #[tokio::test]
    async fn shadow_exposes_user_skills_plugins_rules() {
        // Skills, plugins, and execpolicy rules all live under
        // `$CODEX_HOME` and are missed if not explicitly symlinked.
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        let skill_path = real.path().join("skills/my-skill/SKILL.md");
        let plugin_path = real.path().join("plugins/my-plugin/manifest.json");
        let rule_path = real.path().join("rules/default.rules");
        for p in [&skill_path, &plugin_path, &rule_path] {
            tokio::fs::create_dir_all(p.parent().unwrap())
                .await
                .unwrap();
        }
        tokio::fs::write(&skill_path, b"skill").await.unwrap();
        tokio::fs::write(&plugin_path, b"plugin").await.unwrap();
        tokio::fs::write(&rule_path, b"rule").await.unwrap();

        let shadow = CodexHomeShadow::create_with_real_home(&c, Some(real.path().to_path_buf()))
            .await
            .unwrap();

        assert_eq!(
            tokio::fs::read(shadow.path().join("skills/my-skill/SKILL.md"))
                .await
                .unwrap(),
            b"skill"
        );
        assert_eq!(
            tokio::fs::read(shadow.path().join("plugins/my-plugin/manifest.json"))
                .await
                .unwrap(),
            b"plugin"
        );
        assert_eq!(
            tokio::fs::read(shadow.path().join("rules/default.rules"))
                .await
                .unwrap(),
            b"rule"
        );
    }

    #[tokio::test]
    async fn shadow_exposes_installation_id_and_global_state() {
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        tokio::fs::write(real.path().join("installation_id"), b"abc")
            .await
            .unwrap();
        tokio::fs::write(real.path().join("models_cache.json"), b"{}")
            .await
            .unwrap();
        tokio::fs::write(real.path().join(".codex-global-state.json"), b"{\"x\":1}")
            .await
            .unwrap();

        let shadow = CodexHomeShadow::create_with_real_home(&c, Some(real.path().to_path_buf()))
            .await
            .unwrap();

        assert_eq!(
            tokio::fs::read(shadow.path().join("installation_id"))
                .await
                .unwrap(),
            b"abc"
        );
        assert!(shadow.path().join("models_cache.json").exists());
        assert_eq!(
            tokio::fs::read(shadow.path().join(".codex-global-state.json"))
                .await
                .unwrap(),
            b"{\"x\":1}"
        );
    }

    #[tokio::test]
    async fn missing_real_history_does_not_block_creation() {
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        // No history.jsonl, no sessions/ — link_session_state should
        // create sessions/ and skip history.jsonl, both silently.
        let shadow = CodexHomeShadow::create_with_real_home(&c, Some(real.path().to_path_buf()))
            .await
            .unwrap();
        assert!(shadow.auth_path().exists());
        assert!(!shadow.path().join("history.jsonl").exists());
    }

    #[tokio::test]
    async fn persistent_shadow_seeds_config_once_and_survives_drop() {
        let c = sample_cred();
        let real = tempfile::tempdir().unwrap();
        let config = tempfile::tempdir().unwrap();
        tokio::fs::write(real.path().join("config.toml"), b"model = \"real\"\n")
            .await
            .unwrap();

        let shadow = CodexHomeShadow::create_persistent_with_real_home(
            Some(&c),
            config.path(),
            "key/one",
            Some(real.path().to_path_buf()),
        )
        .await
        .unwrap();
        let path = shadow.path().to_path_buf();
        assert!(path.join("auth.json").exists());
        assert_eq!(
            tokio::fs::read_to_string(path.join("config.toml"))
                .await
                .unwrap(),
            "model = \"real\"\n"
        );

        tokio::fs::write(path.join("config.toml"), b"model = \"app-copy\"\n")
            .await
            .unwrap();
        drop(shadow);
        assert!(path.exists(), "persistent codex-app home must outlive drop");

        let shadow2 = CodexHomeShadow::create_persistent_with_real_home(
            Some(&c),
            config.path(),
            "key/one",
            Some(real.path().to_path_buf()),
        )
        .await
        .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(shadow2.path().join("config.toml"))
                .await
                .unwrap(),
            "model = \"app-copy\"\n"
        );
    }

    #[tokio::test]
    async fn persistent_shadow_without_auth_removes_stale_auth() {
        let c = sample_cred();
        let config = tempfile::tempdir().unwrap();
        let shadow = CodexHomeShadow::create_persistent_with_real_home(
            Some(&c),
            config.path(),
            "key-one",
            None,
        )
        .await
        .unwrap();
        let path = shadow.path().to_path_buf();
        assert!(path.join("auth.json").exists());
        drop(shadow);

        let shadow =
            CodexHomeShadow::create_persistent_with_real_home(None, config.path(), "key-one", None)
                .await
                .unwrap();
        assert!(!shadow.path().join("auth.json").exists());
    }
}
