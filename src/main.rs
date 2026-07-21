//! Main entry point for the aivo CLI.
//!
//! The actual dispatch lives in `aivo::run::run` so internal helpers can stay
//! `pub(crate)`. Keep this file a thin wrapper.
//!
//! We run the tokio current-thread runtime on a dedicated worker thread with an
//! 8 MiB stack. Windows' default main-thread stack is ~1 MiB, which is not
//! enough for `aivo::run::run`'s async state machine after v0.21.0 — bare
//! `aivo --version` overflowed the stack on the Windows CI runner. Linux and
//! macOS default to 8 MiB; pinning the worker explicitly normalizes that
//! across platforms.
fn main() {
    // Wrapper copies of aivo.exe re-exec the bundled codex here, before
    // paying for the worker thread and tokio runtime.
    #[cfg(windows)]
    aivo::services::codex_app_wrapper::maybe_run_windows_shim();

    let worker = std::thread::Builder::new()
        .name("aivo-main".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime");
            rt.block_on(aivo::run::run());
        })
        .expect("failed to spawn aivo-main worker thread");
    worker.join().expect("aivo-main worker thread panicked");
}
