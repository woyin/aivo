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
        for message in &self.queued_messages {
            let recall = message_recall_text(&message.text);
            rows.push(if recall.trim().is_empty() {
                QueuedRow {
                    segment: QueueSegment::Message,
                    display: attachment_summary(&message.attachments),
                    recall,
                }
            } else {
                queued_row(QueueSegment::Message, recall)
            });
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
        let mut recalls: Vec<String> = rows.into_iter().map(|row| row.recall).collect();
        // One tag sequence over recalled messages first, the draft below after.
        let messages = std::mem::take(&mut self.queued_messages);
        let first_message = recalls.len() - messages.len();
        let mut attachments = Vec::new();
        for (recall, message) in recalls[first_message..].iter_mut().zip(messages) {
            *recall = shift_attachment_tags(recall, &message.attachments, attachments.len());
            attachments.extend(message.attachments);
        }
        let shift = attachments.len();
        if shift > 0 {
            let draft_attachments = std::mem::take(&mut self.draft_attachments);
            // Descending, or a rewritten tag collides with the next one.
            for (i, att) in draft_attachments.iter().enumerate().rev() {
                self.replace_in_draft(
                    &attachment_tag(att, i + 1),
                    &attachment_tag(att, shift + i + 1),
                );
            }
            attachments.extend(draft_attachments);
            self.draft_attachments = attachments;
        }
        let recalled = recalls
            .iter()
            .map(String::as_str)
            .filter(|recall| !recall.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        self.queued_commands.clear();
        self.clear_steering_queue();
        self.leave_history_navigation();
        if recalled.is_empty() {
            return true;
        }
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

fn shift_attachment_tags(text: &str, attachments: &[MessageAttachment], offset: usize) -> String {
    let mut text = text.to_string();
    if offset == 0 {
        return text;
    }
    for (i, att) in attachments.iter().enumerate().rev() {
        let old = attachment_tag(att, i + 1);
        if let Some(start) = text.find(&old) {
            text.replace_range(
                start..start + old.len(),
                &attachment_tag(att, offset + i + 1),
            );
        }
    }
    text
}

fn attachment_summary(attachments: &[MessageAttachment]) -> String {
    attachments
        .iter()
        .map(|a| format!("[{}] {}", attachment_kind_label(a), a.name))
        .collect::<Vec<_>>()
        .join(", ")
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
