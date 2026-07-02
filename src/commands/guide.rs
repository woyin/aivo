//! `aivo guide` — prints the embedded usage guide, so the `aivo chat` agent (and users)
//! can answer aivo how-to questions offline instead of fetching docs from the web.

pub const GUIDE: &str = include_str!("aivo_guide.md");

pub fn print_guide() {
    print!("{GUIDE}");
    if !GUIDE.ends_with('\n') {
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::GUIDE;

    #[test]
    fn guide_covers_the_core_surface() {
        // A non-empty guide that names the pieces the agent is most asked about, so
        // it can answer aivo how-to questions offline instead of fetching docs.
        assert!(GUIDE.len() > 500);
        for needle in [
            "aivo keys add",
            "aivo models",
            "aivo chat",
            "/model",
            "/effort",
        ] {
            assert!(GUIDE.contains(needle), "guide should mention `{needle}`");
        }
    }
}
