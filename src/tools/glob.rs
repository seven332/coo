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
        "Fast file pattern matching tool. Returns matching file paths sorted by \
         modification time (most recent first).\n\n\
         - Supports glob patterns like \"**/*.rs\" or \"src/**/*.ts\".\n\
         - Use this when you need to find files by name or extension.\n\
         - For searching file contents, use grep instead."
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
                    "description": "Directory to search in. Omit to use the current working directory."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn call(&self, input: serde_json::Value, context: &super::ToolContext) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        let base = input
            .path
            .map(|p| context.resolve_path(&p))
            .or_else(|| context.cwd.as_ref().map(std::path::PathBuf::from));

        let full_pattern = match &base {
            Some(p) => {
                let p = p.to_string_lossy();
                let p = p.trim_end_matches('/');
                format!("{p}/{}", input.pattern)
            }
            None => input.pattern,
        };

        let paths = match glob::glob(&full_pattern) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid glob pattern: {e}")),
        };

        let mut entries: Vec<(std::path::PathBuf, std::time::SystemTime)> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        for entry in paths {
            match entry {
                Ok(path) => {
                    let mtime = std::fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    entries.push((path, mtime));
                }
                Err(e) => errors.push(format!("(error: {e})")),
            }
        }
        // Sort by modification time, most recent first.
        entries.sort_by(|a, b| b.1.cmp(&a.1));

        let mut matches: Vec<String> = entries
            .into_iter()
            .map(|(p, _)| p.display().to_string())
            .collect();
        matches.extend(errors);

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
                json!({"pattern": "*.rs", "path": "/tmp/nonexistent_coo_glob_test"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert!(text_of(&result).contains("No matches"));
    }

    #[tokio::test]
    async fn cwd_used_when_no_path() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "").unwrap();
        fs::write(dir.path().join("b.txt"), "").unwrap();

        let tool = GlobTool;
        let mut ctx = crate::tools::dummy_context();
        ctx.cwd = Some(dir.path().to_str().unwrap().to_string());
        let result = tool.call(json!({"pattern": "*.rs"}), &ctx).await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("a.rs"));
        assert!(!text.contains("b.txt"));
    }
}
