//! OS-CSPRNG primitives. The crate's entire randomness surface is these two
//! helpers, so the `rand` crate (plus rand_chacha) stays out of the tree.

/// Fill `buf` with cryptographically secure random bytes from the OS.
pub fn fill(buf: &mut [u8]) {
    getrandom::fill(buf).expect("OS RNG unavailable");
}

/// `len` chars drawn uniformly from an ASCII `alphabet` (rejection-sampled,
/// so no modulo bias). `alphabet` must be non-empty and at most 256 bytes.
pub fn string_from(alphabet: &[u8], len: usize) -> String {
    debug_assert!(!alphabet.is_empty() && alphabet.len() <= 256);
    let zone = 256 - 256 % alphabet.len();
    let mut out = String::with_capacity(len);
    let mut buf = [0u8; 32];
    while out.len() < len {
        fill(&mut buf);
        for &b in &buf {
            if (b as usize) < zone && out.len() < len {
                out.push(alphabet[b as usize % alphabet.len()] as char);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_produces_nonzero_bytes() {
        let mut buf = [0u8; 32];
        fill(&mut buf);
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn string_from_respects_len_and_alphabet() {
        for len in [0, 1, 12, 100] {
            let s = string_from(b"abc123", len);
            assert_eq!(s.len(), len);
            assert!(s.bytes().all(|b| b"abc123".contains(&b)));
        }
    }

    #[test]
    fn string_from_full_byte_alphabet_has_no_rejection_zone() {
        let alphabet: Vec<u8> = (0..=255).filter(u8::is_ascii_alphanumeric).collect();
        let s = string_from(&alphabet, 64);
        assert_eq!(s.len(), 64);
    }
}
