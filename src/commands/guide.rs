//! `aivo guide` — prints the embedded usage guide, so the `aivo code` agent (and users)
//! can answer aivo how-to questions offline instead of fetching docs from the web.

pub fn guide() -> &'static str {
    crate::services::embedded_assets::aivo_guide_md()
}

pub fn print_guide() {
    let guide = guide();
    print!("{guide}");
    if !guide.ends_with('\n') {
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::guide;

    #[test]
    fn guide_covers_the_core_surface() {
        // A non-empty guide that names the pieces the agent is most asked about, so
        // it can answer aivo how-to questions offline instead of fetching docs.
        let guide = guide();
        assert!(guide.len() > 500);
        for needle in [
            "aivo keys add",
            "aivo models",
            "aivo code",
            "/model",
            "Ctrl+T",
        ] {
            assert!(guide.contains(needle), "guide should mention `{needle}`");
        }
    }
}
