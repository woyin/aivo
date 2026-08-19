//! Opt-in structured trace, the debugging substrate for render/turn bugs:
//! `AIVO_TRACE_LOG=<path>` appends `<millis> [scope] event=<name> k=v` lines,
//! `AIVO_TRACE_SCOPES=render,turn` filters (unset = all scopes). Tracing can
//! never fail a run: init or write errors silence the tracer for the rest of
//! the process. Disabled costs one `OnceLock` load per call site.

use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub struct Tracer {
    file: Mutex<Option<File>>,
    /// `None` = all scopes.
    scopes: Option<Vec<String>>,
    start: Instant,
}

static TRACER: OnceLock<Option<Tracer>> = OnceLock::new();

fn global() -> Option<&'static Tracer> {
    TRACER
        .get_or_init(|| {
            let path = std::env::var("AIVO_TRACE_LOG").ok()?;
            let scopes = std::env::var("AIVO_TRACE_SCOPES").ok();
            Tracer::create(&path, scopes.as_deref())
        })
        .as_ref()
}

/// True when `scope` is traced — gate `format!` cost on this at call sites.
pub fn enabled(scope: &str) -> bool {
    global().is_some_and(|t| t.wants(scope))
}

pub fn line(scope: &str, msg: &str) {
    if let Some(t) = global() {
        t.line(scope, msg);
    }
}

/// `trace_ev!("render", "event=draw us={}", n)` — formats only when enabled.
#[macro_export]
macro_rules! trace_ev {
    ($scope:literal, $($arg:tt)*) => {
        if $crate::services::trace::enabled($scope) {
            $crate::services::trace::line($scope, &format!($($arg)*));
        }
    };
}

impl Tracer {
    fn create(path: &str, scopes: Option<&str>) -> Option<Tracer> {
        if path.is_empty() {
            return None;
        }
        let file = File::options().create(true).append(true).open(path).ok()?;
        let scopes = scopes
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
        Some(Tracer {
            file: Mutex::new(Some(file)),
            scopes,
            start: Instant::now(),
        })
    }

    fn wants(&self, scope: &str) -> bool {
        self.scopes
            .as_ref()
            .is_none_or(|s| s.iter().any(|x| x == scope))
    }

    fn line(&self, scope: &str, msg: &str) {
        if !self.wants(scope) {
            return;
        }
        let ms = self.start.elapsed().as_millis();
        let mut guard = self.file.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(f) = guard.as_mut()
            && writeln!(f, "{ms} [{scope}] {msg}").is_err()
        {
            *guard = None; // first failure silences the tracer
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(path: &std::path::Path) -> String {
        std::fs::read_to_string(path).unwrap_or_default()
    }

    #[test]
    fn writes_scoped_lines_and_filters() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.log");
        let t = Tracer::create(path.to_str().unwrap(), Some("render, turn")).unwrap();
        t.line("render", "event=draw us=42");
        t.line("mcp", "event=ignored");
        t.line("turn", "event=step n=1");
        let log = read(&path);
        assert!(log.contains("[render] event=draw us=42"), "{log}");
        assert!(log.contains("[turn] event=step n=1"));
        assert!(!log.contains("ignored"));
        assert!(t.wants("render") && !t.wants("mcp"));
    }

    #[test]
    fn no_scopes_means_all() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.log");
        let t = Tracer::create(path.to_str().unwrap(), None).unwrap();
        t.line("anything", "event=x");
        assert!(read(&path).contains("[anything] event=x"));
    }

    #[test]
    fn empty_path_or_unwritable_disables() {
        assert!(Tracer::create("", None).is_none());
        assert!(Tracer::create("/nonexistent-dir-zz/t.log", None).is_none());
    }
}
