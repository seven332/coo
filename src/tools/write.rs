use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::message::ToolResult;

use super::Tool;

pub struct WriteTool;

#[derive(Deserialize)]
struct Input {
    file_path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        })
    }

    async fn call(&self, input: serde_json::Value, _context: &super::ToolContext) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        // Ensure parent directory exists.
        if let Some(parent) = std::path::Path::new(&input.file_path).parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            return ToolResult::error(format!("Failed to create directory: {e}"));
        }

        match tokio::fs::write(&input.file_path, &input.content).await {
            Ok(()) => ToolResult::success(format!("Written to {}", input.file_path)),
            Err(e) => ToolResult::error(format!("Failed to write {}: {e}", input.file_path)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn write_and_verify() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");

        let tool = WriteTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "content": "hello world"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a/b/c/test.txt");

        let tool = WriteTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "content": "nested"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "nested");
    }
}
