use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::message::ToolResult;

use super::Tool;

pub struct GlobTool;

#[derive(Deserialize)]
struct Input {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Returns matching file paths, one per line."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g. \"**/*.rs\", \"src/**/*.ts\")"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn call(&self, input: serde_json::Value, _context: &super::ToolContext) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        let full_pattern = match &input.path {
            Some(p) => {
                let p = p.trim_end_matches('/');
                format!("{p}/{}", input.pattern)
            }
            None => input.pattern,
        };

        let paths = match glob::glob(&full_pattern) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid glob pattern: {e}")),
        };

        let mut matches: Vec<String> = Vec::new();
        for entry in paths {
            match entry {
                Ok(path) => matches.push(path.display().to_string()),
                Err(e) => matches.push(format!("(error: {e})")),
            }
        }

        if matches.is_empty() {
            ToolResult::success("No matches found.")
        } else {
            ToolResult::success(matches.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    fn text_of(result: &crate::message::ToolResult) -> &str {
        match &result.content[0] {
            crate::message::ToolResultContent::Text { text } => text,
        }
    }

    #[tokio::test]
    async fn find_rust_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("foo.rs"), "").unwrap();
        fs::write(dir.path().join("bar.rs"), "").unwrap();
        fs::write(dir.path().join("baz.txt"), "").unwrap();

        let tool = GlobTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap()}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("foo.rs"));
        assert!(text.contains("bar.rs"));
        assert!(!text.contains("baz.txt"));
    }

    #[tokio::test]
    async fn recursive_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("a/b");
        fs::create_dir_all(&sub).unwrap();
        fs::write(dir.path().join("top.rs"), "").unwrap();
        fs::write(sub.join("deep.rs"), "").unwrap();

        let tool = GlobTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({"pattern": "**/*.rs", "path": dir.path().to_str().unwrap()}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("top.rs"));
        assert!(text.contains("deep.rs"));
    }

    #[tokio::test]
    async fn no_matches() {
        let dir = tempfile::tempdir().unwrap();

        let tool = GlobTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({"pattern": "*.xyz", "path": dir.path().to_str().unwrap()}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert!(text_of(&result).contains("No matches"));
    }

    #[tokio::test]
    async fn invalid_pattern() {
        let tool = GlobTool;
        let ctx = crate::tools::dummy_context();
        let result = tool.call(json!({"pattern": "[invalid"}), &ctx).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn nonexistent_path() {
        let tool = GlobTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({"pattern": "*.rs", "path": "/tmp/nonexistent_you_mind_glob_test"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert!(text_of(&result).contains("No matches"));
    }
}
