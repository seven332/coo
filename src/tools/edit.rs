use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::message::ToolResult;

use super::Tool;

pub struct EditTool;

#[derive(Deserialize)]
struct Input {
    file_path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Perform exact string replacements in a file. The old_string must match exactly. \
         Use replace_all to replace every occurrence."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false, replaces first only)"
                }
            },
            "required": ["file_path", "old_string", "new_string"]
        })
    }

    async fn call(&self, input: serde_json::Value) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        if input.old_string == input.new_string {
            return ToolResult::error("old_string and new_string must be different");
        }

        let content = match tokio::fs::read_to_string(&input.file_path).await {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to read {}: {e}", input.file_path)),
        };

        let count = content.matches(&input.old_string).count();

        if count == 0 {
            return ToolResult::error(format!("old_string not found in {}", input.file_path));
        }

        if !input.replace_all && count > 1 {
            return ToolResult::error(format!(
                "old_string has {count} matches in {}. Use replace_all or provide a more unique string.",
                input.file_path
            ));
        }

        let new_content = if input.replace_all {
            content.replace(&input.old_string, &input.new_string)
        } else {
            content.replacen(&input.old_string, &input.new_string, 1)
        };

        match tokio::fs::write(&input.file_path, &new_content).await {
            Ok(()) => ToolResult::success(format!(
                "Replaced {count} occurrence{} in {}",
                if count > 1 { "s" } else { "" },
                input.file_path
            )),
            Err(e) => ToolResult::error(format!("Failed to write {}: {e}", input.file_path)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text_of(result: &crate::message::ToolResult) -> &str {
        match &result.content[0] {
            crate::message::ToolResultContent::Text { text } => text,
        }
    }

    #[tokio::test]
    async fn replace_single() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world").unwrap();

        let tool = EditTool;
        let result = tool
            .call(json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "hello",
                "new_string": "goodbye"
            }))
            .await;
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "goodbye world");
    }

    #[tokio::test]
    async fn replace_all() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "aaa bbb aaa").unwrap();

        let tool = EditTool;
        let result = tool
            .call(json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "aaa",
                "new_string": "ccc",
                "replace_all": true
            }))
            .await;
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "ccc bbb ccc");
        assert!(text_of(&result).contains("2 occurrences"));
    }

    #[tokio::test]
    async fn ambiguous_match_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "foo bar foo").unwrap();

        let tool = EditTool;
        let result = tool
            .call(json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "foo",
                "new_string": "baz"
            }))
            .await;
        assert!(result.is_error);
        assert!(text_of(&result).contains("2 matches"));
        // File should be unchanged.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "foo bar foo");
    }

    #[tokio::test]
    async fn not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").unwrap();

        let tool = EditTool;
        let result = tool
            .call(json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "xyz",
                "new_string": "abc"
            }))
            .await;
        assert!(result.is_error);
        assert!(text_of(&result).contains("not found"));
    }

    #[tokio::test]
    async fn same_string_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").unwrap();

        let tool = EditTool;
        let result = tool
            .call(json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "hello",
                "new_string": "hello"
            }))
            .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn nonexistent_file() {
        let tool = EditTool;
        let result = tool
            .call(json!({
                "file_path": "/tmp/nonexistent_you_mind_edit_test",
                "old_string": "a",
                "new_string": "b"
            }))
            .await;
        assert!(result.is_error);
    }
}
