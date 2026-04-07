use std::sync::atomic::Ordering;
use std::time::Instant;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;
use tracing::info;

use crate::message::{ToolResult, new_uuid};

use super::Tool;

pub struct BashTool;

#[derive(Deserialize)]
struct Input {
    command: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default = "default_timeout")]
    timeout: u64,
    #[serde(default)]
    run_in_background: bool,
}

fn default_timeout() -> u64 {
    120
}

/// Execute a bash command with timeout. Returns (output_text, is_error).
async fn run_command(
    command: &str,
    timeout: std::time::Duration,
    cwd: Option<&str>,
) -> (String, bool) {
    let mut cmd = Command::new("bash");
    cmd.arg("-c").arg(command).kill_on_drop(true);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }

    match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let mut text = String::new();
            if !stdout.is_empty() {
                text.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str("stderr:\n");
                text.push_str(&stderr);
            }
            if text.is_empty() {
                text.push_str("(no output)");
            }
            if output.status.success() {
                (text, false)
            } else {
                (
                    format!("Exit code: {}\n{text}", output.status.code().unwrap_or(-1)),
                    true,
                )
            }
        }
        Ok(Err(e)) => (format!("Failed to execute: {e}"), true),
        Err(_) => ("Command timed out".to_string(), true),
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command and return its output."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "description": {
                    "type": "string",
                    "description": "Clear, concise description of what this command does in active voice. For simple commands (git, npm, standard CLI tools), keep it brief (5-10 words). For complex commands (piped commands, obscure flags), add enough context to clarify."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120)"
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Run the command in the background. Returns immediately. You will be automatically notified when it completes — do NOT sleep or poll."
                }
            },
            "required": ["command"]
        })
    }

    async fn call(&self, input: serde_json::Value, context: &super::ToolContext) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        let description = input.description.unwrap_or_default();

        if input.run_in_background {
            let task_id = new_uuid();
            let timeout = std::time::Duration::from_secs(input.timeout);
            let cwd = context.cwd.clone();
            let background_tx = context.background_tx.clone();
            let pending = context.pending_background.clone();
            let bg_task_id = task_id.clone();
            let bg_description = description.clone();
            let command = input.command;

            info!(task_id, description, "Running bash in background");
            pending.fetch_add(1, Ordering::SeqCst);

            let bg_handle = tokio::spawn(async move {
                let _pending_guard = super::PendingGuard(pending);
                let start = Instant::now();
                let (result, is_error) = run_command(&command, timeout, cwd.as_deref()).await;
                let duration_ms = start.elapsed().as_millis() as u64;

                let _ = background_tx
                    .send(super::BackgroundAgentResult {
                        agent_id: bg_task_id,
                        description: bg_description,
                        result,
                        is_error,
                        duration_ms,
                        total_tokens: 0,
                        tool_use_count: 0,
                    })
                    .await;
            });

            context.background_handles.lock().await.push(bg_handle);

            ToolResult::success(format!(
                "Command started in background.\ntaskId: {task_id}\ndescription: {description}"
            ))
        } else {
            let timeout = std::time::Duration::from_secs(input.timeout);
            let (result, is_error) =
                run_command(&input.command, timeout, context.cwd.as_deref()).await;
            if is_error {
                ToolResult::error(result)
            } else {
                ToolResult::success(result)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_context() -> (
        crate::tools::ToolContext,
        tokio::sync::mpsc::Receiver<crate::tools::BackgroundAgentResult>,
    ) {
        use std::sync::Arc;
        let (background_tx, background_rx) = tokio::sync::mpsc::channel(16);
        let mut ctx = crate::tools::dummy_context();
        ctx.background_tx = background_tx;
        ctx.pending_background = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        ctx.background_handles = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        (ctx, background_rx)
    }

    #[tokio::test]
    async fn echo_command() {
        let tool = BashTool;
        let ctx = crate::tools::dummy_context();
        let result = tool.call(json!({"command": "echo hello"}), &ctx).await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            crate::message::ToolResultContent::Text { text } => text.as_str(),
        };
        assert_eq!(text.trim(), "hello");
    }

    #[tokio::test]
    async fn failing_command() {
        let tool = BashTool;
        let ctx = crate::tools::dummy_context();
        let result = tool.call(json!({"command": "exit 1"}), &ctx).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn invalid_input() {
        let tool = BashTool;
        let ctx = crate::tools::dummy_context();
        let result = tool.call(json!({}), &ctx).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn timeout() {
        let tool = BashTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(json!({"command": "sleep 10", "timeout": 1}), &ctx)
            .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn captures_stderr() {
        let tool = BashTool;
        let ctx = crate::tools::dummy_context();
        let result = tool.call(json!({"command": "echo err >&2"}), &ctx).await;
        let text = match &result.content[0] {
            crate::message::ToolResultContent::Text { text } => text.clone(),
        };
        assert!(text.contains("stderr:"));
        assert!(text.contains("err"));
    }

    #[tokio::test]
    async fn no_output_command() {
        let tool = BashTool;
        let ctx = crate::tools::dummy_context();
        let result = tool.call(json!({"command": "true"}), &ctx).await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            crate::message::ToolResultContent::Text { text } => text.as_str(),
        };
        assert_eq!(text, "(no output)");
    }

    #[tokio::test]
    async fn cwd_respected() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let tool = BashTool;
        let mut ctx = crate::tools::dummy_context();
        ctx.cwd = Some(canonical.to_str().unwrap().to_string());
        let result = tool.call(json!({"command": "pwd"}), &ctx).await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            crate::message::ToolResultContent::Text { text } => text.as_str(),
        };
        assert_eq!(text.trim(), canonical.to_str().unwrap());
    }

    #[tokio::test]
    async fn run_in_background_returns_immediately() {
        let (ctx, mut bg_rx) = make_context();
        let tool = BashTool;
        let result = tool
            .call(
                json!({"command": "echo bg-hello", "run_in_background": true, "description": "test bg"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            crate::message::ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("Command started in background"));
        assert!(text.contains("taskId:"));

        let bg_result = tokio::time::timeout(std::time::Duration::from_secs(5), bg_rx.recv())
            .await
            .expect("should complete within timeout")
            .expect("should receive result");
        assert!(!bg_result.is_error);
        assert!(bg_result.result.contains("bg-hello"));
        assert_eq!(bg_result.description, "test bg");
        assert_eq!(bg_result.total_tokens, 0);
        assert_eq!(bg_result.tool_use_count, 0);
    }

    #[tokio::test]
    async fn run_in_background_error_reported() {
        let (ctx, mut bg_rx) = make_context();
        let tool = BashTool;
        let result = tool
            .call(
                json!({"command": "exit 42", "run_in_background": true}),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "should return success (command started)");

        let bg_result = tokio::time::timeout(std::time::Duration::from_secs(5), bg_rx.recv())
            .await
            .expect("should complete within timeout")
            .expect("should receive result");
        assert!(bg_result.is_error);
        assert!(bg_result.result.contains("Exit code: 42"));
    }

    #[tokio::test]
    async fn run_in_background_timeout() {
        let (ctx, mut bg_rx) = make_context();
        let tool = BashTool;
        let result = tool
            .call(
                json!({"command": "sleep 60", "run_in_background": true, "timeout": 1}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);

        let bg_result = tokio::time::timeout(std::time::Duration::from_secs(5), bg_rx.recv())
            .await
            .expect("should complete within timeout")
            .expect("should receive result");
        assert!(bg_result.is_error);
        assert!(bg_result.result.contains("timed out"));
    }
}
