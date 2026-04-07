use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;

use crate::message::ToolResult;

use super::Tool;

pub struct GrepTool;

#[derive(Deserialize)]
struct Input {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default = "default_true")]
    line_numbers: bool,
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default = "default_output_mode")]
    output_mode: String,
}

fn default_true() -> bool {
    true
}

fn default_output_mode() -> String {
    "files_with_matches".to_string()
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents using regex patterns. Uses ripgrep (rg) if available, \
         falls back to grep."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default: current directory)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. \"*.rs\", \"*.{ts,tsx}\")"
                },
                "line_numbers": {
                    "type": "boolean",
                    "description": "Show line numbers (default: true)"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case insensitive search (default: false)"
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output mode: \"content\" shows matching lines, \"files_with_matches\" shows only file paths (default), \"count\" shows match counts per file."
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

        if !matches!(
            input.output_mode.as_str(),
            "content" | "files_with_matches" | "count"
        ) {
            return ToolResult::error(format!(
                "Unknown output_mode: {}. Use content, files_with_matches, or count.",
                input.output_mode
            ));
        }

        let resolved = input
            .path
            .as_deref()
            .map(|p| context.resolve_path(p))
            .or_else(|| context.cwd.as_ref().map(std::path::PathBuf::from));
        let search_path = resolved
            .as_deref()
            .unwrap_or(std::path::Path::new("."))
            .to_string_lossy();

        // Try rg first, fall back to grep.
        let has_rg = Command::new("rg")
            .arg("--version")
            .output()
            .await
            .is_ok_and(|o| o.status.success());

        let output = if has_rg {
            run_rg(&input, &search_path).await
        } else {
            run_grep(&input, &search_path).await
        };

        match output {
            Ok(text) => {
                if text.is_empty() {
                    ToolResult::success("No matches found.")
                } else {
                    ToolResult::success(text)
                }
            }
            Err(e) => ToolResult::error(e),
        }
    }
}

async fn run_rg(input: &Input, path: &str) -> Result<String, String> {
    let mut cmd = Command::new("rg");
    cmd.arg("--no-heading");

    match input.output_mode.as_str() {
        "files_with_matches" => {
            cmd.arg("-l");
        }
        "count" => {
            cmd.arg("-c");
        }
        _ => {
            // "content" mode: show matching lines.
            if input.line_numbers {
                cmd.arg("-n");
            }
        }
    }

    if input.case_insensitive {
        cmd.arg("-i");
    }
    if let Some(ref g) = input.glob {
        cmd.arg("--glob").arg(g);
    }

    cmd.arg(&input.pattern).arg(path);

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run rg: {e}"))?;

    // rg exits 1 for no matches, 2 for errors.
    if output.status.code() == Some(2) {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("rg error: {stderr}"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

async fn run_grep(input: &Input, path: &str) -> Result<String, String> {
    let mut cmd = Command::new("grep");
    cmd.arg("-r").arg("-E");

    match input.output_mode.as_str() {
        "files_with_matches" => {
            cmd.arg("-l");
        }
        "count" => {
            cmd.arg("-c");
        }
        _ => {
            if input.line_numbers {
                cmd.arg("-n");
            }
        }
    }

    if input.case_insensitive {
        cmd.arg("-i");
    }
    if let Some(ref g) = input.glob {
        cmd.arg("--include").arg(g);
    }

    cmd.arg(&input.pattern).arg(path);

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run grep: {e}"))?;

    // grep exits 1 for no matches, 2 for errors.
    if output.status.code() == Some(2) {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("grep error: {stderr}"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
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
    async fn search_in_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again",
        )
        .unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "hello",
                    "path": dir.path().join("test.txt").to_str().unwrap(),
                    "output_mode": "content"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("hello world"));
        assert!(text.contains("hello again"));
        assert!(!text.contains("foo bar"));
    }

    #[tokio::test]
    async fn search_in_directory() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "match_me here").unwrap();
        fs::write(dir.path().join("b.txt"), "nothing here").unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "match_me",
                    "path": dir.path().to_str().unwrap(),
                    "output_mode": "content"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("match_me"));
        assert!(text.contains("a.txt"));
    }

    #[tokio::test]
    async fn no_matches() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.txt"), "hello").unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "nonexistent_xyz",
                    "path": dir.path().to_str().unwrap()
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert!(text_of(&result).contains("No matches"));
    }

    #[tokio::test]
    async fn case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.txt"), "Hello World").unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "hello",
                    "path": dir.path().join("test.txt").to_str().unwrap(),
                    "case_insensitive": true,
                    "output_mode": "content"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert!(text_of(&result).contains("Hello World"));
    }

    #[tokio::test]
    async fn with_line_numbers() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.txt"), "aaa\nbbb\nccc").unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "bbb",
                    "path": dir.path().join("test.txt").to_str().unwrap(),
                    "output_mode": "content"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("2:") || text.contains("2-"));
    }

    #[tokio::test]
    async fn invalid_regex() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.txt"), "hello").unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "[invalid",
                    "path": dir.path().to_str().unwrap()
                }),
                &ctx,
            )
            .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn files_with_matches_mode() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "target line").unwrap();
        fs::write(dir.path().join("b.txt"), "no match").unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "target",
                    "path": dir.path().to_str().unwrap()
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("a.txt"));
        assert!(!text.contains("b.txt"));
        // Should not contain the matched line content.
        assert!(!text.contains("target line"));
    }

    #[tokio::test]
    async fn count_mode() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.txt"), "aaa\nbbb\naaa").unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "aaa",
                    "path": dir.path().join("test.txt").to_str().unwrap(),
                    "output_mode": "count"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("2"), "should show count of 2, got: {text}");
    }

    #[tokio::test]
    async fn unknown_output_mode_rejected() {
        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(json!({"pattern": "x", "output_mode": "invalid"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(text_of(&result).contains("Unknown output_mode"));
    }

    #[tokio::test]
    async fn cwd_used_when_no_path() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file.txt"), "findme here").unwrap();

        let tool = GrepTool;
        let mut ctx = crate::tools::dummy_context();
        ctx.cwd = Some(dir.path().to_str().unwrap().to_string());
        let result = tool.call(json!({"pattern": "findme"}), &ctx).await;
        assert!(!result.is_error);
        let text = text_of(&result);
        // Default output_mode is files_with_matches, so result is file paths.
        assert!(text.contains("file.txt"));
    }
}
