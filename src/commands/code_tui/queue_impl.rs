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
        for (offset, text) in steering.iter().enumerate() {
            rows.push(queued_row(
                QueueSegment::Steering,
                offset,
                message_recall_text(text),
            ));
        }
        for (offset, command) in self.queued_commands.iter().enumerate() {
            rows.push(queued_row(
                QueueSegment::Command,
                offset,
                command_recall_text(command),
            ));
        }
        for (offset, text) in self.queued_messages.iter().enumerate() {
            rows.push(queued_row(
                QueueSegment::Message,
                offset,
                message_recall_text(text),
            ));
        }
        rows
    }

    /// Remove the row from its owning queue; `false` = the engine drained it.
    pub(super) fn queue_row_remove(&mut self, row: &QueuedRow) -> bool {
        match row.segment {
            QueueSegment::Steering => {
                let mut queue = self
                    .steering_queue
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match validated_position(&queue, row, |s| message_recall_text(s)) {
                    Some(pos) => {
                        queue.remove(pos);
                        true
                    }
                    None => false,
                }
            }
            QueueSegment::Command => {
                match validated_position(&self.queued_commands, row, command_recall_text) {
                    Some(pos) => {
                        self.queued_commands.remove(pos);
                        true
                    }
                    None => false,
                }
            }
            QueueSegment::Message => {
                match validated_position(&self.queued_messages, row, |s| message_recall_text(s)) {
                    Some(pos) => {
                        self.queued_messages.remove(pos);
                        true
                    }
                    None => false,
                }
            }
        }
    }

    /// Remove the row and hand back the text to re-edit in the composer.
    pub(super) fn queue_row_recall(&mut self, row: &QueuedRow) -> Option<String> {
        self.queue_row_remove(row).then(|| row.recall.clone())
    }

    /// Swap the row with its neighbor toward `dir` (−1 earlier, +1 later);
    /// within its own segment only — delivery semantics differ across segments.
    pub(super) fn queue_row_move(&mut self, row: &QueuedRow, dir: i8) -> bool {
        match row.segment {
            QueueSegment::Steering => {
                let mut queue = self
                    .steering_queue
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match validated_position(&queue, row, |s| message_recall_text(s)) {
                    Some(pos) => swap_neighbor(&mut queue, pos, dir),
                    None => false,
                }
            }
            QueueSegment::Command => {
                match validated_position(&self.queued_commands, row, command_recall_text) {
                    Some(pos) => swap_neighbor(&mut self.queued_commands, pos, dir),
                    None => false,
                }
            }
            QueueSegment::Message => {
                match validated_position(&self.queued_messages, row, |s| message_recall_text(s)) {
                    Some(pos) => swap_neighbor(&mut self.queued_messages, pos, dir),
                    None => false,
                }
            }
        }
    }
}

/// Compact `/name args` for an expanded skill body (re-expands on resubmit),
/// else the raw text.
fn message_recall_text(text: &str) -> String {
    skill_invocation_label(text).unwrap_or_else(|| text.to_string())
}

fn queued_row(segment: QueueSegment, offset: usize, recall: String) -> QueuedRow {
    let display = recall.replace('\n', " ⏎ ");
    QueuedRow {
        segment,
        offset,
        display,
        recall,
    }
}

/// The row's current position: the snapshotted offset when it still matches,
/// else by recall text; `None` = the engine consumed it.
fn validated_position<T>(
    items: &[T],
    row: &QueuedRow,
    recall_of: impl Fn(&T) -> String,
) -> Option<usize> {
    if items
        .get(row.offset)
        .is_some_and(|item| recall_of(item) == row.recall)
    {
        return Some(row.offset);
    }
    items.iter().position(|item| recall_of(item) == row.recall)
}

fn swap_neighbor<T>(items: &mut [T], pos: usize, dir: i8) -> bool {
    let target = if dir < 0 {
        pos.checked_sub(1)
    } else {
        Some(pos + 1)
    };
    match target {
        Some(target) if target < items.len() => {
            items.swap(pos, target);
            true
        }
        _ => false,
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
        SlashCommand::Attach(path) => format!("/attach {path}"),
        SlashCommand::Detach(n) => format!("/detach {n}"),
        SlashCommand::Copy(None) => "/copy".to_string(),
        SlashCommand::Copy(Some(n)) => format!("/copy {n}"),
        SlashCommand::Skills(arg) => with_arg("skills", arg),
        SlashCommand::Agents(arg) => with_arg("agents", arg),
        SlashCommand::Mcp(arg) => with_arg("mcp", arg),
        SlashCommand::Goal(arg) => with_arg("goal", arg),
        SlashCommand::Plan(arg) => with_arg("plan", arg),
        SlashCommand::Review(arg) => with_arg("review", arg),
        SlashCommand::Memory => "/memory".to_string(),
        SlashCommand::Effort(arg) => with_arg("effort", arg),
        SlashCommand::CreateSkill(arg) => with_arg("create-skill", arg),
        SlashCommand::Skill { name, argument } => with_arg(name, argument),
        SlashCommand::Rewind => "/rewind".to_string(),
        SlashCommand::Config => "/config".to_string(),
        SlashCommand::Compact { fast: true } => "/compact fast".to_string(),
        SlashCommand::Compact { fast: false } => "/compact".to_string(),
        SlashCommand::Context => "/context".to_string(),
        SlashCommand::Share(arg) => with_arg("share", arg),
        SlashCommand::Login => "/login".to_string(),
        SlashCommand::Logout => "/logout".to_string(),
        SlashCommand::Usage => "/usage".to_string(),
        SlashCommand::Help => "/help".to_string(),
    }
}
