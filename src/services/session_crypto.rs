use aes::Aes256;
use aes_gcm::{
    AesGcm,
    aead::{Aead, KeyInit, consts::U16, generic_array::GenericArray},
};
use anyhow::Result;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use pbkdf2::pbkdf2_hmac;
use rand::RngCore;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::services::{os_keyring, system_env};

pub const ENCRYPTION_MARKER: &str = "enc:";
pub const V3_ENCRYPTION_MARKER: &str = "enc3:";
pub const V4_ENCRYPTION_MARKER: &str = "enc4:";
pub const V5_ENCRYPTION_MARKER: &str = "enc5:";

const IV_LENGTH: usize = 16;
const SALT_LENGTH: usize = 32;
const KEY_LENGTH: usize = 32;

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
struct SecretKey([u8; KEY_LENGTH]);

impl SecretKey {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

#[cfg(any(test, feature = "__internal_test_fast_crypto"))]
const ITERATIONS: u32 = 100;
#[cfg(not(any(test, feature = "__internal_test_fast_crypto")))]
const ITERATIONS: u32 = 100_000;

fn derive_key() -> SecretKey {
    static CACHED_KEY: OnceLock<SecretKey> = OnceLock::new();
    CACHED_KEY.get_or_init(derive_key_inner).clone()
}

fn derive_key_inner() -> SecretKey {
    let username = system_env::username().unwrap_or_default();
    let homedir: String = system_env::home_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let machine_data = format!("{}:{}", username, homedir);

    let mut hasher = Sha256::new();
    hasher.update(b"aivo-salt");
    hasher.update(machine_data.as_bytes());
    let salt_full = hasher.finalize();
    let salt = &salt_full[..SALT_LENGTH];

    let mut key = [0u8; KEY_LENGTH];
    pbkdf2_hmac::<Sha256>(machine_data.as_bytes(), salt, ITERATIONS, &mut key);

    SecretKey(key)
}

fn derive_key_v3() -> SecretKey {
    static CACHED_KEY: OnceLock<SecretKey> = OnceLock::new();
    CACHED_KEY.get_or_init(derive_key_v3_inner).clone()
}

fn derive_key_v3_inner() -> SecretKey {
    let username = system_env::username().unwrap_or_default();
    let homedir: String = system_env::home_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    // Use legacy machine_id to preserve backward compatibility with existing v3 keys
    let machine_id = system_env::machine_id_legacy().unwrap_or_default();
    let machine_data = format!("{}:{}:{}", username, homedir, machine_id);

    let mut hasher = Sha256::new();
    hasher.update(b"aivo-salt-v3");
    hasher.update(machine_data.as_bytes());
    let salt_full = hasher.finalize();
    let salt = &salt_full[..SALT_LENGTH];

    let mut key = [0u8; KEY_LENGTH];
    pbkdf2_hmac::<Sha256>(machine_data.as_bytes(), salt, ITERATIONS, &mut key);

    SecretKey(key)
}

fn derive_key_v4() -> SecretKey {
    static CACHED_KEY: OnceLock<SecretKey> = OnceLock::new();
    CACHED_KEY.get_or_init(derive_key_v4_inner).clone()
}

fn derive_key_v4_inner() -> SecretKey {
    let username = system_env::username().unwrap_or_default();
    let homedir: String = system_env::home_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let machine_id = system_env::machine_id().unwrap_or_default();
    let machine_data = format!("{}:{}:{}", username, homedir, machine_id);

    let mut hasher = Sha256::new();
    hasher.update(b"aivo-salt-v4");
    hasher.update(machine_data.as_bytes());
    let salt_full = hasher.finalize();
    let salt = &salt_full[..SALT_LENGTH];

    let mut key = [0u8; KEY_LENGTH];
    pbkdf2_hmac::<Sha256>(machine_data.as_bytes(), salt, ITERATIONS, &mut key);

    SecretKey(key)
}

fn derive_key_v5(secret: &os_keyring::MasterSecret) -> SecretKey {
    // The keyring secret is full-entropy random; a domain-separated hash
    // replaces PBKDF2 (stretching adds nothing to a 256-bit random input).
    let mut hasher = Sha256::new();
    hasher.update(b"aivo-key-v5");
    hasher.update(secret.as_slice());
    SecretKey(hasher.finalize().into())
}

/// v5 write key when the OS keyring is opted in and usable; creates the
/// master secret on first use. None falls back to v4 (today's behavior).
fn v5_write_key() -> Option<SecretKey> {
    if !os_keyring::keyring_enabled() {
        return None;
    }
    os_keyring::ensure_master_secret().map(|s| derive_key_v5(&s))
}

pub fn is_encrypted(value: &str) -> bool {
    value.starts_with(V5_ENCRYPTION_MARKER)
        || value.starts_with(V4_ENCRYPTION_MARKER)
        || value.starts_with(V3_ENCRYPTION_MARKER)
        || value.starts_with(ENCRYPTION_MARKER)
}

/// Returns true if the value uses the version `encrypt` would write today
/// (v5 when the OS keyring is active, otherwise v4). v5 values are always
/// current so disabling the keyring never downgrades them.
pub fn is_current_encryption(value: &str) -> bool {
    if value.starts_with(V5_ENCRYPTION_MARKER) {
        return true;
    }
    value.starts_with(V4_ENCRYPTION_MARKER) && v5_write_key().is_none()
}

type Aes256Gcm16 = AesGcm<Aes256, U16, U16>;

pub fn encrypt(plaintext: &str) -> Result<String> {
    if plaintext.is_empty() {
        return Ok(plaintext.to_string());
    }

    if is_encrypted(plaintext) {
        return Ok(plaintext.to_string());
    }

    let (key, marker) = match v5_write_key() {
        Some(key) => (key, V5_ENCRYPTION_MARKER),
        None => (derive_key_v4(), V4_ENCRYPTION_MARKER),
    };
    let cipher = Aes256Gcm16::new(GenericArray::from_slice(key.as_slice()));

    let mut iv = [0u8; IV_LENGTH];
    rand::thread_rng().fill_bytes(&mut iv);

    let nonce = GenericArray::from_slice(&iv);
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_bytes())
        .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

    let mut combined = Vec::with_capacity(IV_LENGTH + ciphertext.len());
    combined.extend_from_slice(&iv);
    combined.extend_from_slice(&ciphertext);

    Ok(format!("{}{}", marker, BASE64.encode(&combined)))
}

pub fn decrypt(encrypted_data: &str) -> Result<String> {
    if encrypted_data.is_empty() {
        return Ok(encrypted_data.to_string());
    }

    if !is_encrypted(encrypted_data) {
        return Err(anyhow::anyhow!("Invalid encrypted data: missing marker"));
    }

    let (key, marker_len) = if encrypted_data.starts_with(V5_ENCRYPTION_MARKER) {
        // Reads ignore the AIVO_KEYCHAIN opt-in gate: opting out must never
        // brick existing v5 values.
        let secret = os_keyring::master_secret().ok_or_else(|| {
            anyhow::Error::new(crate::errors::CLIError::new(
                "Decryption failed - master secret not found in the OS keychain/keyring",
                crate::errors::ErrorCategory::Auth,
                Some(
                    "this value was encrypted with a keyring-held secret (service \"aivo\") \
                     that is no longer readable",
                ),
                Some(
                    "re-add the key, or move keys between machines with \
                     `aivo keys export` / `aivo keys import`",
                ),
            ))
        })?;
        (derive_key_v5(&secret), V5_ENCRYPTION_MARKER.len())
    } else if encrypted_data.starts_with(V4_ENCRYPTION_MARKER) {
        (derive_key_v4(), V4_ENCRYPTION_MARKER.len())
    } else if encrypted_data.starts_with(V3_ENCRYPTION_MARKER) {
        (derive_key_v3(), V3_ENCRYPTION_MARKER.len())
    } else {
        (derive_key(), ENCRYPTION_MARKER.len())
    };

    let cipher = Aes256Gcm16::new(GenericArray::from_slice(key.as_slice()));

    let data = BASE64
        .decode(&encrypted_data[marker_len..])
        .map_err(|e| anyhow::anyhow!("Base64 decode failed: {}", e))?;

    if data.len() < IV_LENGTH {
        return Err(anyhow::anyhow!("Invalid encrypted data: too short"));
    }

    let iv = &data[..IV_LENGTH];
    let ciphertext = &data[IV_LENGTH..];
    let nonce = GenericArray::from_slice(iv);

    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| anyhow::anyhow!("Decryption failed - key may be from different machine"))?;

    String::from_utf8(plaintext)
        .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in decrypted data: {}", e))
}

#[cfg(test)]
mod tests {
    use super::{
        Aes256Gcm16, ENCRYPTION_MARKER, IV_LENGTH, V3_ENCRYPTION_MARKER, V4_ENCRYPTION_MARKER,
        V5_ENCRYPTION_MARKER, decrypt, derive_key, derive_key_v3, encrypt, is_current_encryption,
        is_encrypted,
    };
    use crate::services::os_keyring;
    use aes_gcm::aead::{Aead, KeyInit, generic_array::GenericArray};
    use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
    use rand::RngCore;

    #[test]
    fn test_encryption_format() {
        let plaintext = "test-api-key-12345";
        let encrypted = encrypt(plaintext).unwrap();

        assert!(encrypted.starts_with(V4_ENCRYPTION_MARKER));

        let data = &encrypted[V4_ENCRYPTION_MARKER.len()..];
        let decoded = BASE64.decode(data).unwrap();
        assert!(decoded.len() >= 32);
    }

    #[test]
    fn test_encryption_roundtrip() {
        let test_cases = [
            "simple-key",
            "key-with-special-chars-!@#$%",
            "sk-ant-api03-test123",
            "unicode-キー-测试",
        ];

        for plaintext in test_cases {
            let encrypted = encrypt(plaintext).unwrap();
            let decrypted = decrypt(&encrypted).unwrap();
            assert_eq!(decrypted, plaintext);
        }
    }

    #[test]
    fn test_is_encrypted_detection() {
        assert!(is_encrypted("enc:abc123"));
        assert!(is_encrypted("enc3:abc123"));
        assert!(is_encrypted("enc4:abc123"));
        assert!(is_encrypted("enc5:abc123"));
        assert!(!is_encrypted("plain-text"));
        assert!(!is_encrypted(""));
        assert!(!is_encrypted("enc"));
    }

    #[test]
    fn test_is_current_encryption() {
        assert!(is_current_encryption("enc5:abc123"));
        assert!(is_current_encryption("enc4:abc123"));
        assert!(!is_current_encryption("enc3:abc123"));
        assert!(!is_current_encryption("enc:abc123"));
        assert!(!is_current_encryption("plain-text"));
    }

    #[test]
    fn test_v5_roundtrip_with_keyring() {
        os_keyring::test_state::set(true, Some([7u8; 32]));
        let encrypted = encrypt("v5-api-key").unwrap();
        assert!(encrypted.starts_with(V5_ENCRYPTION_MARKER));
        assert_eq!(decrypt(&encrypted).unwrap(), "v5-api-key");
    }

    #[test]
    fn test_v5_requires_opt_in() {
        os_keyring::test_state::set(false, Some([7u8; 32]));
        let encrypted = encrypt("still-v4").unwrap();
        assert!(encrypted.starts_with(V4_ENCRYPTION_MARKER));
    }

    #[test]
    fn test_v5_falls_back_without_secret() {
        os_keyring::test_state::set(true, None);
        let encrypted = encrypt("still-v4").unwrap();
        assert!(encrypted.starts_with(V4_ENCRYPTION_MARKER));
        // v4 must stay "current" so migration doesn't churn on keyring-less hosts
        assert!(is_current_encryption(&encrypted));
    }

    #[test]
    fn test_v4_to_v5_migration_roundtrip() {
        os_keyring::test_state::set(false, None);
        let v4_encrypted = encrypt("migrate-me-v4-to-v5").unwrap();
        assert!(v4_encrypted.starts_with(V4_ENCRYPTION_MARKER));

        os_keyring::test_state::set(true, Some([9u8; 32]));
        assert!(!is_current_encryption(&v4_encrypted));

        let decrypted = decrypt(&v4_encrypted).unwrap();
        let v5_encrypted = encrypt(&decrypted).unwrap();
        assert!(v5_encrypted.starts_with(V5_ENCRYPTION_MARKER));
        assert!(is_current_encryption(&v5_encrypted));
        assert_eq!(decrypt(&v5_encrypted).unwrap(), "migrate-me-v4-to-v5");
    }

    #[test]
    fn test_v5_decrypt_without_secret_errors() {
        os_keyring::test_state::set(true, Some([3u8; 32]));
        let v5_encrypted = encrypt("locked-out").unwrap();

        os_keyring::test_state::set(false, None);
        let err = decrypt(&v5_encrypted).unwrap_err().to_string();
        assert!(err.contains("master secret"));
    }

    #[test]
    fn test_v5_never_downgrades_when_keyring_disabled() {
        os_keyring::test_state::set(true, Some([4u8; 32]));
        let v5_encrypted = encrypt("no-downgrade").unwrap();

        os_keyring::test_state::set(false, None);
        assert!(is_current_encryption(&v5_encrypted));
        let re_encrypted = encrypt(&v5_encrypted).unwrap();
        assert_eq!(re_encrypted, v5_encrypted);
    }

    #[test]
    fn test_legacy_v2_decrypt() {
        let plaintext = "legacy-api-key-v2";
        let key = derive_key();
        let cipher = Aes256Gcm16::new(GenericArray::from_slice(key.as_slice()));

        let mut iv = [0u8; IV_LENGTH];
        rand::thread_rng().fill_bytes(&mut iv);
        let nonce = GenericArray::from_slice(&iv);
        let ciphertext = cipher.encrypt(nonce, plaintext.as_bytes()).unwrap();

        let mut combined = Vec::with_capacity(IV_LENGTH + ciphertext.len());
        combined.extend_from_slice(&iv);
        combined.extend_from_slice(&ciphertext);

        let v2_encrypted = format!("{}{}", ENCRYPTION_MARKER, BASE64.encode(&combined));
        let decrypted = decrypt(&v2_encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_legacy_v3_decrypt() {
        let plaintext = "legacy-api-key-v3";
        let key = derive_key_v3();
        let cipher = Aes256Gcm16::new(GenericArray::from_slice(key.as_slice()));

        let mut iv = [0u8; IV_LENGTH];
        rand::thread_rng().fill_bytes(&mut iv);
        let nonce = GenericArray::from_slice(&iv);
        let ciphertext = cipher.encrypt(nonce, plaintext.as_bytes()).unwrap();

        let mut combined = Vec::with_capacity(IV_LENGTH + ciphertext.len());
        combined.extend_from_slice(&iv);
        combined.extend_from_slice(&ciphertext);

        let v3_encrypted = format!("{}{}", V3_ENCRYPTION_MARKER, BASE64.encode(&combined));
        let decrypted = decrypt(&v3_encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_v3_to_v4_migration_roundtrip() {
        let plaintext = "migrate-me-v3-to-v4";
        let key = derive_key_v3();
        let cipher = Aes256Gcm16::new(GenericArray::from_slice(key.as_slice()));

        let mut iv = [0u8; IV_LENGTH];
        rand::thread_rng().fill_bytes(&mut iv);
        let nonce = GenericArray::from_slice(&iv);
        let ciphertext = cipher.encrypt(nonce, plaintext.as_bytes()).unwrap();

        let mut combined = Vec::with_capacity(IV_LENGTH + ciphertext.len());
        combined.extend_from_slice(&iv);
        combined.extend_from_slice(&ciphertext);
        let v3_encrypted = format!("{}{}", V3_ENCRYPTION_MARKER, BASE64.encode(&combined));

        // Decrypt old v3 key
        let decrypted = decrypt(&v3_encrypted).unwrap();
        assert_eq!(decrypted, plaintext);

        // Re-encrypt with v4
        let v4_encrypted = encrypt(&decrypted).unwrap();
        assert!(v4_encrypted.starts_with(V4_ENCRYPTION_MARKER));

        // Verify v4 roundtrip
        let decrypted_v4 = decrypt(&v4_encrypted).unwrap();
        assert_eq!(decrypted_v4, plaintext);
    }

    #[test]
    fn test_encryption_never_panics() {
        let inputs = [
            "a",
            "normal-key",
            "key-with-symbols!@#",
            "sk-test123456789",
            "unicode-キー-测试",
        ];

        for input in inputs {
            let encrypted = encrypt(input).expect("encryption should not fail");
            assert!(is_encrypted(&encrypted));

            let decrypted = decrypt(&encrypted).expect("decryption should not fail");
            assert_eq!(decrypted, input);
        }

        assert_eq!(encrypt("").unwrap(), "");
    }

    #[test]
    fn test_double_encryption_idempotent() {
        let key = "my-api-key";
        let encrypted1 = encrypt(key).unwrap();
        let encrypted2 = encrypt(&encrypted1).unwrap();

        assert_eq!(encrypted1, encrypted2);
    }
}
