/**
 * Main entry point for the aivo CLI.
 *
 * The actual dispatch lives in `aivo::run::run` so internal helpers can stay
 * `pub(crate)`. Keep this file a thin wrapper.
 */
#[tokio::main(flavor = "current_thread")]
async fn main() {
    aivo::run::run().await;
}
