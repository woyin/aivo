use super::*;

impl CodeTuiApp {
    /// Snapshot the three pending queues as one row list in delivery order:
    /// steering → commands → messages.
    pub(super) fn queued_rows(&self) -> Vec<QueuedRow> {
        let steering: Vec<String> = self
            .steering_queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let mut rows = Vec::with_capacity(
            steering.len() + self.queued_commands.len() + self.queued_messages.len(),
        );
        for text in &steering {
            rows.push(queued_row(
                QueueSegment::Steering,
                message_recall_text(text),
            ));
        }
        for command in &self.queued_commands {
            rows.push(queued_row(
                QueueSegment::Command,
                command_recall_text(command),
            ));
        }
        for text in &self.queued_messages {
            rows.push(queued_row(QueueSegment::Message, message_recall_text(text)));
        }
        rows
    }

    /// Drain every pending row into the composer as one newline-joined draft,
    /// keeping typed text on its own line below. `false` = nothing was queued.
    pub(super) fn recall_queued_into_draft(&mut self) -> bool {
        let rows = self.queued_rows();
        if rows.is_empty() {
            return false;
        }
        let recalled = rows
            .iter()
            .map(|row| row.recall.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        self.queued_messages.clear();
        self.queued_commands.clear();
        self.clear_steering_queue();
        self.leave_history_navigation();
        if self.draft.is_empty() {
            self.cursor = recalled.len();
            self.draft = recalled;
        } else {
            self.cursor += recalled.len() + 1;
            self.draft = format!("{recalled}\n{}", self.draft);
        }
        true
    }
}

/// Compact `/name args` for an expanded skill body (re-expands on resubmit),
/// else the raw text.
fn message_recall_text(text: &str) -> String {
    skill_invocation_label(text).unwrap_or_else(|| text.to_string())
}

fn queued_row(segment: QueueSegment, recall: String) -> QueuedRow {
    let display = recall.replace('\n', " ⏎ ");
    QueuedRow {
        segment,
        display,
        recall,
    }
}

/// Reverse of `parse_slash_command`; total so a new variant can't silently
/// queue without a recallable form.
pub(super) fn command_recall_text(command: &SlashCommand) -> String {
    fn with_arg(name: &str, argument: &Option<String>) -> String {
        match argument {
            Some(arg) => format!("/{name} {arg}"),
            None => format!("/{name}"),
        }
    }
    match command {
        SlashCommand::New => "/new".to_string(),
        SlashCommand::Exit => "/exit".to_string(),
        SlashCommand::Resume(arg) => with_arg("resume", arg),
        SlashCommand::Model(arg) => with_arg("model", arg),
        SlashCommand::Key(arg) => with_arg("key", arg),
        SlashCommand::Copy(None) => "/copy".to_string(),
        SlashCommand::Copy(Some(n)) => format!("/copy {n}"),
        SlashCommand::Preview(None) => "/preview".to_string(),
        SlashCommand::Preview(Some(target)) => format!("/preview {target}"),
        SlashCommand::Skills(arg) => with_arg("skills", arg),
        SlashCommand::Agents(arg) => with_arg("agents", arg),
        SlashCommand::Mcp(arg) => with_arg("mcp", arg),
        SlashCommand::Goal(arg) => with_arg("goal", arg),
        SlashCommand::Plan(arg) => with_arg("plan", arg),
        SlashCommand::Ask(arg) => with_arg("ask", arg),
        SlashCommand::Btw(arg) => with_arg("btw", arg),
        SlashCommand::CreateSkill(arg) => with_arg("create-skill", arg),
        SlashCommand::Skill { name, argument } => with_arg(name, argument),
        SlashCommand::Rewind => "/rewind".to_string(),
        SlashCommand::Config => "/config".to_string(),
        SlashCommand::Compact { fast: true } => "/compact fast".to_string(),
        SlashCommand::Compact { fast: false } => "/compact".to_string(),
        SlashCommand::Context => "/context".to_string(),
        SlashCommand::Session => "/session".to_string(),
        SlashCommand::Share(arg) => with_arg("share", arg),
        SlashCommand::Login => "/login".to_string(),
        SlashCommand::Logout => "/logout".to_string(),
        SlashCommand::Usage => "/usage".to_string(),
        SlashCommand::Help => "/help".to_string(),
    }
}
