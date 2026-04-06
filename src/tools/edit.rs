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

    async fn call(&self, input: serde_json::Value, context: &super::ToolContext) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        if input.old_string.is_empty() {
            return ToolResult::error("old_string must not be empty");
        }

        if input.old_string == input.new_string {
            return ToolResult::error("old_string and new_string must be different");
        }

        let path = context.resolve_path(&input.file_path);

        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to read {}: {e}", path.display())),
        };

        let count = content.matches(&input.old_string).count();

        if count == 0 {
            return ToolResult::error(format!("old_string not found in {}", path.display()));
        }

        if !input.replace_all && count > 1 {
            return ToolResult::error(format!(
                "old_string has {count} matches in {}. Use replace_all or provide a more unique string.",
                path.display()
            ));
        }

        let new_content = if input.replace_all {
            content.replace(&input.old_string, &input.new_string)
        } else {
            content.replacen(&input.old_string, &input.new_string, 1)
        };

        match tokio::fs::write(&path, &new_content).await {
            Ok(()) => ToolResult::success(format!(
                "Replaced {count} occurrence{} in {}",
                if count > 1 { "s" } else { "" },
                path.display()
            )),
            Err(e) => ToolResult::error(format!("Failed to write {}: {e}", path.display())),
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
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "hello",
                    "new_string": "goodbye"
                }),
                &ctx,
            )
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
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "aaa",
                    "new_string": "ccc",
                    "replace_all": true
                }),
                &ctx,
            )
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
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "foo",
                    "new_string": "baz"
                }),
                &ctx,
            )
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
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "xyz",
                    "new_string": "abc"
                }),
                &ctx,
            )
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
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "hello",
                    "new_string": "hello"
                }),
                &ctx,
            )
            .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn nonexistent_file() {
        let tool = EditTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": "/tmp/nonexistent_coo_edit_test",
                    "old_string": "a",
                    "new_string": "b"
                }),
                &ctx,
            )
            .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn empty_old_string_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").unwrap();

        let tool = EditTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "",
                    "new_string": "x"
                }),
                &ctx,
            )
            .await;
        assert!(result.is_error);
        assert!(text_of(&result).contains("empty"));
    }

    #[tokio::test]
    async fn multiline_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line1\nline2\nline3").unwrap();

        let tool = EditTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "line1\nline2",
                    "new_string": "replaced"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "replaced\nline3");
    }

    #[tokio::test]
    async fn edit_relative_with_cwd() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("target.txt"), "old content").unwrap();

        let tool = EditTool;
        let mut ctx = crate::tools::dummy_context();
        ctx.cwd = Some(dir.path().to_str().unwrap().to_string());
        let result = tool
            .call(
                json!({
                    "file_path": "target.txt",
                    "old_string": "old content",
                    "new_string": "new content"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("target.txt")).unwrap(),
            "new content"
        );
    }
}
