//! Password-encrypted envelope for portable key exports.
//!
//! Unlike `session_crypto`, which derives its key from machine-bound
//! identifiers (username/home/machine_id) so the on-disk config is tied to one
//! host, this module derives the key from a user-chosen password and a random
//! per-export salt. The envelope can be safely moved between machines and
//! decrypted on any host that knows the password.
//!
//! ## On-disk format
//!
//! ```json
//! { "version": 1, "salt": "<base64>", "iv": "<base64>", "ciphertext": "<base64>" }
//! ```
//!
//! Every algorithm choice is implied by `version`. v1 = Argon2id (m=64 MiB,
//! t=3, p=1) + AES-256-GCM with a 16-byte tag. Future format changes bump
//! the version; aivo dispatches on the integer.

use aes::Aes256;
use aes_gcm::{
    AesGcm,
    aead::{Aead, KeyInit, Payload, consts::U16, generic_array::GenericArray},
};
use anyhow::{Result, anyhow};
use argon2::{Algorithm, Argon2, Params, Version};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

pub const EXPORT_VERSION: u32 = 1;

const SALT_LENGTH: usize = 32;
const IV_LENGTH: usize = 16;
const KEY_LENGTH: usize = 32;

/// 64 MiB memory matches 1Password / libsodium "moderate"; tests use the
/// crate minimum to stay fast.
#[cfg(any(test, feature = "__internal_test_fast_crypto"))]
const V1_ARGON2_MEMORY_KIB: u32 = 8;
#[cfg(not(any(test, feature = "__internal_test_fast_crypto")))]
const V1_ARGON2_MEMORY_KIB: u32 = 65_536;
#[cfg(any(test, feature = "__internal_test_fast_crypto"))]
const V1_ARGON2_TIME_COST: u32 = 1;
#[cfg(not(any(test, feature = "__internal_test_fast_crypto")))]
const V1_ARGON2_TIME_COST: u32 = 3;
const V1_ARGON2_PARALLELISM: u32 = 1;

type Aes256Gcm16 = AesGcm<Aes256, U16, U16>;

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
struct DerivedKey([u8; KEY_LENGTH]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportEnvelope {
    pub version: u32,
    pub salt: String,
    pub iv: String,
    pub ciphertext: String,
}

impl ExportEnvelope {
    /// AAD bound to the GCM tag. Includes `version` so an attacker can't
    /// graft a future-format header onto v1 ciphertext.
    fn aad(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut push = |bytes: &[u8]| {
            out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
            out.extend_from_slice(bytes);
        };
        push(&self.version.to_be_bytes());
        push(self.salt.as_bytes());
        push(self.iv.as_bytes());
        out
    }

    /// Cheap structural check; does NOT authenticate the payload.
    pub fn validate_header(&self) -> Result<()> {
        if self.version != EXPORT_VERSION {
            return Err(anyhow!(
                "Unsupported export version {} (this build expects {})",
                self.version,
                EXPORT_VERSION
            ));
        }
        Ok(())
    }
}

fn derive_key_v1(password: &str, salt: &[u8]) -> Result<DerivedKey> {
    let mut key = [0u8; KEY_LENGTH];
    let params = Params::new(
        V1_ARGON2_MEMORY_KIB,
        V1_ARGON2_TIME_COST,
        V1_ARGON2_PARALLELISM,
        Some(KEY_LENGTH),
    )
    .map_err(|e| anyhow!("invalid Argon2id parameters: {}", e))?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    argon
        .hash_password_into(password.as_bytes(), salt, &mut key)
        .map_err(|e| anyhow!("Argon2id key derivation failed: {}", e))?;
    Ok(DerivedKey(key))
}

pub fn encrypt_export(plaintext: &[u8], password: &str) -> Result<ExportEnvelope> {
    if password.is_empty() {
        return Err(anyhow!("Password must not be empty"));
    }

    let mut salt = [0u8; SALT_LENGTH];
    rand::thread_rng().fill_bytes(&mut salt);

    let mut iv = [0u8; IV_LENGTH];
    rand::thread_rng().fill_bytes(&mut iv);

    // Build envelope first so the encrypt-time AAD comes from the same
    // string forms `aad()` recomputes on decrypt — no drift risk.
    let envelope_template = ExportEnvelope {
        version: EXPORT_VERSION,
        salt: BASE64.encode(salt),
        iv: BASE64.encode(iv),
        ciphertext: String::new(),
    };
    let aad = envelope_template.aad();

    let key = derive_key_v1(password, &salt)?;
    let cipher = Aes256Gcm16::new(GenericArray::from_slice(&key.0));
    let nonce = GenericArray::from_slice(&iv);

    let ciphertext = cipher
        .encrypt(
            nonce,
            Payload {
                msg: plaintext,
                aad: &aad,
            },
        )
        .map_err(|e| anyhow!("Export encryption failed: {}", e))?;

    let mut envelope = envelope_template;
    envelope.ciphertext = BASE64.encode(&ciphertext);
    Ok(envelope)
}

/// Wrong password, tampered header, and corrupted ciphertext all surface
/// the same generic error to avoid oracle leaks.
pub fn decrypt_export(envelope: &ExportEnvelope, password: &str) -> Result<Vec<u8>> {
    envelope.validate_header()?;

    if password.is_empty() {
        return Err(anyhow!("Password must not be empty"));
    }

    let salt = BASE64
        .decode(&envelope.salt)
        .map_err(|_| anyhow!("Invalid export file: salt is not valid base64"))?;
    if salt.len() != SALT_LENGTH {
        return Err(anyhow!("Invalid export file: unexpected salt length"));
    }

    let iv = BASE64
        .decode(&envelope.iv)
        .map_err(|_| anyhow!("Invalid export file: iv is not valid base64"))?;
    if iv.len() != IV_LENGTH {
        return Err(anyhow!("Invalid export file: unexpected iv length"));
    }

    let ciphertext = BASE64
        .decode(&envelope.ciphertext)
        .map_err(|_| anyhow!("Invalid export file: ciphertext is not valid base64"))?;

    let aad = envelope.aad();

    let key = match envelope.version {
        1 => derive_key_v1(password, &salt)?,
        v => return Err(anyhow!("Unsupported export version {}", v)),
    };
    let cipher = Aes256Gcm16::new(GenericArray::from_slice(&key.0));
    let nonce = GenericArray::from_slice(&iv);

    cipher
        .decrypt(
            nonce,
            Payload {
                msg: ciphertext.as_slice(),
                aad: &aad,
            },
        )
        .map_err(|_| anyhow!("Decryption failed — wrong password or corrupted file"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_recovers_plaintext() {
        let plaintext = br#"[{"id":"abc","key":"sk-test"}]"#;
        let env = encrypt_export(plaintext, "correct horse battery staple").unwrap();
        let decrypted = decrypt_export(&env, "correct horse battery staple").unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn wrong_password_rejected() {
        let env = encrypt_export(b"secret", "right-pass").unwrap();
        let err = decrypt_export(&env, "wrong-pass").unwrap_err();
        assert!(
            err.to_string().contains("Decryption failed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn empty_password_rejected_on_encrypt() {
        let err = encrypt_export(b"x", "").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn empty_password_rejected_on_decrypt() {
        let env = encrypt_export(b"x", "pass").unwrap();
        let err = decrypt_export(&env, "").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn each_export_uses_fresh_salt_and_iv() {
        let a = encrypt_export(b"same input", "same-password").unwrap();
        let b = encrypt_export(b"same input", "same-password").unwrap();
        assert_ne!(a.salt, b.salt);
        assert_ne!(a.iv, b.iv);
        assert_ne!(a.ciphertext, b.ciphertext);
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut env = encrypt_export(b"x", "pass").unwrap();
        env.version = 999;
        let err = decrypt_export(&env, "pass").unwrap_err();
        assert!(err.to_string().contains("Unsupported export version"));
    }

    #[test]
    fn tampered_ciphertext_rejected() {
        let mut env = encrypt_export(b"hello", "pass").unwrap();
        let mut bytes = BASE64.decode(&env.ciphertext).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        env.ciphertext = BASE64.encode(&bytes);
        let err = decrypt_export(&env, "pass").unwrap_err();
        assert!(err.to_string().contains("Decryption failed"));
    }

    #[test]
    fn header_mutation_invalidates_tag() {
        // Every cleartext field is AAD-bound; mutating any of them must
        // prevent decryption (whether via header rejection or GCM tag).
        fn flip_byte_in_b64(s: &str) -> String {
            let mut bytes = BASE64.decode(s).unwrap();
            bytes[0] ^= 0xFF;
            BASE64.encode(&bytes)
        }
        let mutators: &[fn(&mut ExportEnvelope)] = &[
            |e| e.version = 2,
            |e| e.salt = flip_byte_in_b64(&e.salt),
            |e| e.iv = flip_byte_in_b64(&e.iv),
        ];

        for mutate in mutators {
            let mut env = encrypt_export(b"secret payload", "pass").unwrap();
            mutate(&mut env);
            assert!(
                decrypt_export(&env, "pass").is_err(),
                "mutated envelope unexpectedly decrypted"
            );
        }
    }

    #[test]
    fn envelope_serialises_to_json() {
        let env = encrypt_export(b"x", "pass").unwrap();
        let json = serde_json::to_string(&env).unwrap();
        let parsed: ExportEnvelope = serde_json::from_str(&json).unwrap();
        let decrypted = decrypt_export(&parsed, "pass").unwrap();
        assert_eq!(decrypted, b"x");
    }

    #[test]
    fn envelope_has_only_essential_fields() {
        let env = encrypt_export(b"x", "pass").unwrap();
        let json: serde_json::Value = serde_json::to_value(&env).unwrap();
        let obj = json.as_object().unwrap();
        let keys: std::collections::BTreeSet<&str> = obj.keys().map(String::as_str).collect();
        let expected: std::collections::BTreeSet<&str> = ["version", "salt", "iv", "ciphertext"]
            .into_iter()
            .collect();
        assert_eq!(keys, expected, "unexpected envelope fields: {:?}", keys);
    }
}
