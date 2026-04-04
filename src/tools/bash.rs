use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;

use crate::message::ToolResult;

use super::Tool;

pub struct BashTool;

#[derive(Deserialize)]
struct Input {
    command: String,
    #[serde(default = "default_timeout")]
    timeout: u64,
}

fn default_timeout() -> u64 {
    120
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
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120)"
                }
            },
            "required": ["command"]
        })
    }

    async fn call(&self, input: serde_json::Value, _context: &super::ToolContext) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        let timeout = std::time::Duration::from_secs(input.timeout);
        let output_fut = Command::new("bash")
            .arg("-c")
            .arg(&input.command)
            .kill_on_drop(true)
            .output();

        match tokio::time::timeout(timeout, output_fut).await {
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
                    ToolResult::success(text)
                } else {
                    ToolResult::error(format!(
                        "Exit code: {}\n{text}",
                        output.status.code().unwrap_or(-1)
                    ))
                }
            }
            Ok(Err(e)) => ToolResult::error(format!("Failed to execute: {e}")),
            Err(_) => ToolResult::error("Command timed out"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
}
