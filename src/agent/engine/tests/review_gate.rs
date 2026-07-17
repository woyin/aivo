use super::super::*;
use super::helpers::*;
use serde_json::json;

/// A fake UI for the edit-review gate: records the card firings and returns a preset verdict.
#[derive(Default)]
struct ReviewUi {
    reject: bool,
    review_calls: usize,
    reviewed_paths: Vec<String>,
}

impl AgentUi for ReviewUi {
    fn assistant_text(&mut self, _: &str) {}
    fn tool_start(&mut self, _: &str, _: &Value) {}
    fn tool_result(&mut self, _: &str, _: &Result<String, String>) {}
    fn notify(&mut self, _: &str) {}
    fn footer(&mut self, _: Option<&str>, _: usize, _: u64, _: u64, _: u64) {}
    fn ask_permission<'a>(&'a mut self, _: &'a str, _: Option<&'a str>) -> BoxFuture<'a, Decision> {
        Box::pin(async { Decision::Allow })
    }
    fn review_edits<'a>(
        &'a mut self,
        items: &'a [crate::agent::review::ReviewItem],
    ) -> BoxFuture<'a, crate::agent::review::ReviewDecision> {
        self.review_calls += 1;
        for it in items {
            self.reviewed_paths.extend(it.paths.clone());
        }
        let reject = self.reject;
        Box::pin(async move {
            if reject {
                crate::agent::review::ReviewDecision::Reject
            } else {
                crate::agent::review::ReviewDecision::ApproveAll
            }
        })
    }
}

fn write_and_note_calls() -> Vec<ToolCall> {
    vec![
        ToolCall {
            id: "1".into(),
            name: "write_file".into(),
            arguments: json!({"path": "out.txt", "content": "new\n"}),
        },
        ToolCall {
            id: "2".into(),
            name: "take_note".into(),
            arguments: json!({"note": "remember"}),
        },
    ]
}

/// Reject: the edit's outcome is the directive and the file is untouched, while a non-edit sibling still runs.
#[tokio::test]
async fn review_gate_reject_skips_write_but_runs_sibling() {
    use std::sync::atomic::AtomicBool;
    let dir = tmp();
    std::fs::write(dir.join("out.txt"), "old\n").unwrap();
    let client = reqwest::Client::new();
    let flag = AtomicBool::new(true);
    let ctx = TurnCtx {
        client: &client,
        serve_base: "",
        auth: None,
        cwd: dir.as_path(),
        yes: true, // auto-approve on, so no permission card competes with review
        auto_approve_all: false,
        auto_approve: None,
        review_edits: Some(&flag),
    };
    let mut e = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = ReviewUi {
        reject: true,
        ..Default::default()
    };
    let (_t, failures) = e
        .execute_tool_batch(&ctx, &mut ui, &write_and_note_calls())
        .await;
    assert_eq!(ui.review_calls, 1, "the batch is reviewed exactly once");
    assert_eq!(ui.reviewed_paths, vec!["out.txt".to_string()]);
    assert_eq!(
        std::fs::read_to_string(dir.join("out.txt")).unwrap(),
        "old\n",
        "a rejected edit leaves the file untouched"
    );
    assert!(
        failures
            .iter()
            .any(|(t, msg)| t == "write_file"
                && msg == crate::agent::review::REVIEW_REJECTED_DIRECTIVE),
        "write_file reports the review-rejected directive: {failures:?}"
    );
    assert!(
        !failures.iter().any(|(t, _)| t == "take_note"),
        "the non-edit sibling still ran"
    );
}

/// Review ON + ApproveAll: the edit is written.
#[tokio::test]
async fn review_gate_approve_writes_the_edit() {
    use std::sync::atomic::AtomicBool;
    let dir = tmp();
    std::fs::write(dir.join("out.txt"), "old\n").unwrap();
    let client = reqwest::Client::new();
    let flag = AtomicBool::new(true);
    let ctx = TurnCtx {
        client: &client,
        serve_base: "",
        auth: None,
        cwd: dir.as_path(),
        yes: true,
        auto_approve_all: false,
        auto_approve: None,
        review_edits: Some(&flag),
    };
    let mut e = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = ReviewUi::default();
    let (_t, failures) = e
        .execute_tool_batch(&ctx, &mut ui, &write_and_note_calls())
        .await;
    assert_eq!(ui.review_calls, 1);
    assert!(
        failures.is_empty(),
        "approve lets the edit succeed: {failures:?}"
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("out.txt")).unwrap(),
        "new\n",
        "an approved edit is written"
    );
}

/// Flag `None` (headless): the gate is skipped — `review_edits` is never called.
#[tokio::test]
async fn review_gate_none_skips_the_card() {
    let dir = tmp();
    std::fs::write(dir.join("out.txt"), "old\n").unwrap();
    let client = reqwest::Client::new();
    let ctx = TurnCtx {
        client: &client,
        serve_base: "",
        auth: None,
        cwd: dir.as_path(),
        yes: true,
        auto_approve_all: false,
        auto_approve: None,
        review_edits: None,
    };
    let mut e = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = ReviewUi {
        reject: true, // would reject if consulted — proves it isn't
        ..Default::default()
    };
    let (_t, failures) = e
        .execute_tool_batch(&ctx, &mut ui, &write_and_note_calls())
        .await;
    assert_eq!(ui.review_calls, 0, "the gate never fires without the flag");
    assert!(failures.is_empty());
    assert_eq!(
        std::fs::read_to_string(dir.join("out.txt")).unwrap(),
        "new\n"
    );
}

/// With the flag on but no edit tools in the batch, the card doesn't fire.
#[tokio::test]
async fn review_gate_skips_when_no_edits() {
    use std::sync::atomic::AtomicBool;
    let dir = tmp();
    let client = reqwest::Client::new();
    let flag = AtomicBool::new(true);
    let ctx = TurnCtx {
        client: &client,
        serve_base: "",
        auth: None,
        cwd: dir.as_path(),
        yes: true,
        auto_approve_all: false,
        auto_approve: None,
        review_edits: Some(&flag),
    };
    let mut e = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = ReviewUi {
        reject: true,
        ..Default::default()
    };
    let calls = vec![ToolCall {
        id: "1".into(),
        name: "take_note".into(),
        arguments: json!({"note": "just a note"}),
    }];
    let (_t, failures) = e.execute_tool_batch(&ctx, &mut ui, &calls).await;
    assert_eq!(ui.review_calls, 0, "no edit tools → no review card");
    assert!(failures.is_empty());
}
