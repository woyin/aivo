//! Device identity and request signing for the aivo starter endpoint.
//!
//! Ed25519 keypair per install; private seed at `<config>/secrets/device-key`
//! (0600), public key is the `device_id`. Each request is signed over
//! `${device_id}:${timestamp}` so a leaked `device_id` alone can't impersonate.

use ed25519_dalek::{Signer, SigningKey};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::services::http_utils::current_unix_ts;
use crate::version::VERSION;

static SIGNING_KEY: OnceLock<SigningKey> = OnceLock::new();
static DEVICE_ID: OnceLock<String> = OnceLock::new();

const SIGNATURE_ALG: &str = "ed25519";

/// Per-install Ed25519 signing key. Falls back to an ephemeral key on persistence
/// failure rather than reusing the historical shared-secret that collided on Termux.
fn signing_key() -> &'static SigningKey {
    SIGNING_KEY.get_or_init(|| load_or_create_signing_key().unwrap_or_else(generate_signing_key))
}

/// Hex-encoded Ed25519 public key (64 chars).
pub fn device_id() -> &'static str {
    DEVICE_ID.get_or_init(|| {
        let pubkey = signing_key().verifying_key().to_bytes();
        hex_encode(&pubkey)
    })
}

/// Ed25519 signature over `${device_id}:${timestamp}`, lowercase hex (128 chars).
/// The `_device_id` arg is ignored; kept for callsite stability.
pub fn sign_request(_device_id: &str, timestamp: u64) -> String {
    let message = format!("{}:{}", device_id(), timestamp);
    let signature = signing_key().sign(message.as_bytes());
    hex_encode(&signature.to_bytes())
}

/// Conditionally attaches device identity headers when `is_starter` is true.
pub fn maybe_with_starter_headers(
    builder: reqwest::RequestBuilder,
    is_starter: bool,
) -> reqwest::RequestBuilder {
    if is_starter {
        with_starter_headers(builder)
    } else {
        builder
    }
}

/// Device-identity headers as `(name, value)` pairs, for callers that build
/// requests by hand (the `share_tunnel` WS upgrade). [`with_starter_headers`] wraps this.
pub fn starter_header_pairs() -> [(&'static str, String); 5] {
    let did = device_id();
    let ts = current_unix_ts();
    let sig = sign_request(did, ts);
    [
        ("X-Aivo-Device", did.to_string()),
        ("X-Aivo-Timestamp", ts.to_string()),
        ("X-Aivo-Signature", sig),
        ("X-Aivo-Signature-Alg", SIGNATURE_ALG.to_string()),
        ("X-Aivo-Version", VERSION.to_string()),
    ]
}

/// Attaches device identity headers to a request builder.
pub fn with_starter_headers(builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    starter_header_pairs()
        .into_iter()
        .fold(builder, |b, (name, value)| b.header(name, value))
}

pub(crate) fn hex_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn device_key_path() -> Option<PathBuf> {
    Some(crate::services::paths::device_key(
        &crate::services::paths::config_dir(),
    ))
}

fn load_or_create_signing_key() -> Option<SigningKey> {
    let path = device_key_path()?;
    if let Ok(existing) = std::fs::read_to_string(&path)
        && let Some(seed) = parse_seed_hex(existing.trim())
    {
        return Some(SigningKey::from_bytes(&seed));
    }
    let key = generate_signing_key();
    let parent = path.parent()?;
    std::fs::create_dir_all(parent).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
    }
    let seed_hex = hex_encode(&key.to_bytes());
    std::fs::write(&path, &seed_hex).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Some(key)
}

fn generate_signing_key() -> SigningKey {
    let mut seed = [0u8; 32];
    super::rng::fill(&mut seed);
    SigningKey::from_bytes(&seed)
}

fn parse_seed_hex(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Verifier, VerifyingKey};

    // Mirrors the gateway's verification.
    fn verify(device_id: &str, timestamp: u64, signature: &str) -> bool {
        let Some(pubkey_bytes) = parse_seed_hex(device_id) else {
            return false;
        };
        let Ok(vk) = VerifyingKey::from_bytes(&pubkey_bytes) else {
            return false;
        };
        let sig_bytes: [u8; 64] = (0..64)
            .map(|i| u8::from_str_radix(&signature[i * 2..i * 2 + 2], 16).unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
        let message = format!("{}:{}", device_id, timestamp);
        vk.verify(message.as_bytes(), &sig).is_ok()
    }

    #[test]
    fn device_id_is_64_char_hex() {
        let id = device_id();
        assert_eq!(id.len(), 64);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn device_id_is_stable() {
        assert_eq!(device_id(), device_id());
    }

    #[test]
    fn sign_request_produces_128_char_hex() {
        let sig = sign_request(device_id(), 1700000000);
        assert_eq!(sig.len(), 128);
        assert!(sig.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sign_request_verifies_against_device_id_pubkey() {
        let did = device_id();
        let ts = 1700000000;
        let sig = sign_request(did, ts);
        assert!(verify(did, ts, &sig));
    }

    #[test]
    fn sign_request_rejects_wrong_timestamp() {
        let did = device_id();
        let sig = sign_request(did, 1700000000);
        assert!(!verify(did, 1700000001, &sig));
    }

    #[test]
    fn sign_request_rejects_other_device_id() {
        let did = device_id();
        let sig = sign_request(did, 1700000000);
        let other_did = hex_encode(&generate_signing_key().verifying_key().to_bytes());
        assert_ne!(did, other_did);
        assert!(!verify(&other_did, 1700000000, &sig));
    }

    #[test]
    fn generate_signing_key_produces_unique_pubkeys() {
        let a = generate_signing_key().verifying_key().to_bytes();
        let b = generate_signing_key().verifying_key().to_bytes();
        assert_ne!(a, b, "fresh keypairs must not collide");
    }

    #[test]
    fn fresh_keypair_pubkey_is_not_unknown_hash() {
        // sha256("unknown") was the historical Termux-collision fallback.
        let pubkey = generate_signing_key().verifying_key().to_bytes();
        assert_ne!(
            hex_encode(&pubkey),
            "b23a6a8439c0dde5515893e7c90c1e3233b8616e634470f20dc4928bcf3609bc"
        );
    }

    #[test]
    fn parse_seed_hex_round_trips() {
        let key = generate_signing_key();
        let hex = hex_encode(&key.to_bytes());
        let parsed = parse_seed_hex(&hex).expect("round-trip");
        let restored = SigningKey::from_bytes(&parsed);
        assert_eq!(key.to_bytes(), restored.to_bytes());
        assert_eq!(
            key.verifying_key().to_bytes(),
            restored.verifying_key().to_bytes()
        );
    }

    #[test]
    fn parse_seed_hex_rejects_bad_input() {
        assert!(parse_seed_hex("").is_none());
        assert!(parse_seed_hex("zz").is_none());
        assert!(parse_seed_hex(&"a".repeat(63)).is_none());
        assert!(parse_seed_hex(&"a".repeat(65)).is_none());
        assert!(parse_seed_hex(&"g".repeat(64)).is_none());
    }
}
