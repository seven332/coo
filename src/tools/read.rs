use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::message::ToolResult;

use super::Tool;

pub struct ReadTool;

#[derive(Deserialize)]
struct Input {
    file_path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read a file from the filesystem. Returns content with line numbers."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["file_path"]
        })
    }

    async fn call(&self, input: serde_json::Value) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        let content = match tokio::fs::read_to_string(&input.file_path).await {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to read {}: {e}", input.file_path)),
        };

        let lines: Vec<&str> = content.lines().collect();
        let offset = input.offset.unwrap_or(0);
        let limit = input.limit.unwrap_or(2000);
        let end = (offset + limit).min(lines.len());

        if offset >= lines.len() {
            return ToolResult::error(format!(
                "Offset {offset} is past end of file ({} lines)",
                lines.len()
            ));
        }

        let mut output = String::new();
        for (i, line) in lines[offset..end].iter().enumerate() {
            let line_num = offset + i + 1;
            output.push_str(&format!("{line_num}\t{line}\n"));
        }

        ToolResult::success(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write as _;

    fn text_of(result: &crate::message::ToolResult) -> &str {
        match &result.content[0] {
            crate::message::ToolResultContent::Text { text } => text,
        }
    }

    #[tokio::test]
    async fn read_file() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "line1").unwrap();
        writeln!(tmp, "line2").unwrap();
        writeln!(tmp, "line3").unwrap();

        let tool = ReadTool;
        let result = tool
            .call(json!({"file_path": tmp.path().to_str().unwrap()}))
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("1\tline1"));
        assert!(text.contains("3\tline3"));
    }

    #[tokio::test]
    async fn read_with_offset_and_limit() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(tmp, "line{i}").unwrap();
        }

        let tool = ReadTool;
        let result = tool
            .call(json!({
                "file_path": tmp.path().to_str().unwrap(),
                "offset": 2,
                "limit": 3
            }))
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("3\tline3"));
        assert!(text.contains("5\tline5"));
        assert!(!text.contains("6\tline6"));
    }

    #[tokio::test]
    async fn read_nonexistent() {
        let tool = ReadTool;
        let result = tool
            .call(json!({"file_path": "/tmp/nonexistent_you_mind_test_file"}))
            .await;
        assert!(result.is_error);
    }
}
