mod support;

use aivo::services::session_store::{SessionStore, decrypt, encrypt, is_encrypted};
use tempfile::TempDir;

fn store_in_tmpdir(tmp: &TempDir) -> SessionStore {
    let config_path = tmp.path().join("config.json");
    SessionStore::with_path(config_path)
}

#[tokio::test]
async fn add_list_activate_remove() {
    let tmp = TempDir::new().unwrap();
    let store = store_in_tmpdir(&tmp);

    // Start empty
    let keys = store.get_keys().await.unwrap();
    assert!(keys.is_empty());

    // Add two keys
    let id1 = store
        .add_key_with_protocol(
            "anthropic",
            "https://api.anthropic.com/v1",
            None,
            "sk-ant-1",
        )
        .await
        .unwrap();
    let id2 = store
        .add_key_with_protocol("openai", "https://api.openai.com/v1", None, "sk-oai-2")
        .await
        .unwrap();

    assert_ne!(id1, id2);
    assert_eq!(id1.len(), 3);
    assert_eq!(id2.len(), 3);

    // List returns both
    let keys = store.get_keys().await.unwrap();
    assert_eq!(keys.len(), 2);

    // Keys are stored encrypted
    assert!(is_encrypted(&keys[0].key));
    assert!(is_encrypted(&keys[1].key));

    // Set active + verify
    store.set_active_key(&id1).await.unwrap();
    let config = store.load().await.unwrap();
    assert_eq!(config.active_key_id.as_deref(), Some(id1.as_str()));

    // Switch active
    store.set_active_key(&id2).await.unwrap();
    let config = store.load().await.unwrap();
    assert_eq!(config.active_key_id.as_deref(), Some(id2.as_str()));

    // Get key by ID decrypts the secret
    let fetched = store.get_key_by_id(&id1).await.unwrap().unwrap();
    assert_eq!(fetched.name, "anthropic");
    assert_eq!(&*fetched.key, "sk-ant-1");
    assert!(!is_encrypted(&fetched.key));

    // Update key
    let updated = store
        .update_key(
            &id1,
            "anthropic-v2",
            "https://api.anthropic.com/v2",
            None,
            "sk-ant-new",
        )
        .await
        .unwrap();
    assert!(updated);
    let fetched = store.get_key_by_id(&id1).await.unwrap().unwrap();
    assert_eq!(fetched.name, "anthropic-v2");
    assert_eq!(&*fetched.key, "sk-ant-new");

    // Remove key
    let removed = store.delete_key(&id1).await.unwrap();
    assert!(removed);
    let keys = store.get_keys().await.unwrap();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].id, id2);

    // Removing active key clears active_key_id
    store.set_active_key(&id2).await.unwrap();
    store.delete_key(&id2).await.unwrap();
    let config = store.load().await.unwrap();
    assert!(config.active_key_id.is_none());
    assert!(config.api_keys.is_empty());
}

#[tokio::test]
async fn set_active_nonexistent_key_fails() {
    let tmp = TempDir::new().unwrap();
    let store = store_in_tmpdir(&tmp);
    let result = store.set_active_key("nope").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn delete_nonexistent_key_returns_false() {
    let tmp = TempDir::new().unwrap();
    let store = store_in_tmpdir(&tmp);
    let removed = store.delete_key("nope").await.unwrap();
    assert!(!removed);
}

#[tokio::test]
async fn resolve_key_by_name() {
    let tmp = TempDir::new().unwrap();
    let store = store_in_tmpdir(&tmp);

    let id = store
        .add_key_with_protocol("my-provider", "https://example.com/v1", None, "secret-123")
        .await
        .unwrap();

    // Resolve by name
    let key = store
        .resolve_key_by_id_or_name("my-provider")
        .await
        .unwrap();
    assert_eq!(key.id, id);
    assert_eq!(&*key.key, "secret-123");

    // Resolve by short ID
    let key = store.resolve_key_by_id_or_name(&id).await.unwrap();
    assert_eq!(key.name, "my-provider");

    // Resolve nonexistent fails
    let result = store.resolve_key_by_id_or_name("nonexistent").await;
    assert!(result.is_err());
}

#[test]
fn encryption_roundtrip() {
    let secrets = [
        "sk-ant-api03-test123",
        "key-with-special-chars-!@#$%",
        "unicode-\u{30AD}\u{30FC}-\u{6D4B}\u{8BD5}",
    ];
    for secret in secrets {
        let enc = encrypt(secret).unwrap();
        assert!(is_encrypted(&enc));
        assert_ne!(enc, secret);
        let dec = decrypt(&enc).unwrap();
        assert_eq!(dec, secret);
    }
}

#[test]
fn encrypt_empty_is_noop() {
    assert_eq!(encrypt("").unwrap(), "");
    assert_eq!(decrypt("").unwrap(), "");
}

#[test]
fn double_encrypt_is_idempotent() {
    let enc1 = encrypt("my-key").unwrap();
    let enc2 = encrypt(&enc1).unwrap();
    assert_eq!(enc1, enc2);
}
