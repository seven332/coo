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
    #[serde(default = "default_head_limit")]
    head_limit: usize,
    #[serde(default, rename = "-A")]
    after_context: Option<usize>,
    #[serde(default, rename = "-B")]
    before_context: Option<usize>,
    #[serde(default)]
    context: Option<usize>,
    #[serde(default, rename = "type")]
    file_type: Option<String>,
    #[serde(default)]
    multiline: bool,
}

fn default_true() -> bool {
    true
}

fn default_output_mode() -> String {
    "files_with_matches".to_string()
}

fn default_head_limit() -> usize {
    250
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
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Limit output to first N lines/entries (default: 250). Pass 0 for unlimited."
                },
                "-A": {
                    "type": "integer",
                    "description": "Number of lines to show after each match. Requires output_mode: \"content\"."
                },
                "-B": {
                    "type": "integer",
                    "description": "Number of lines to show before each match. Requires output_mode: \"content\"."
                },
                "context": {
                    "type": "integer",
                    "description": "Number of lines to show before and after each match (shorthand for -A and -B). Requires output_mode: \"content\"."
                },
                "type": {
                    "type": "string",
                    "description": "File type to search (rg --type). Common types: js, py, rust, go, java, ts, css, html, json, md."
                },
                "multiline": {
                    "type": "boolean",
                    "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false. Only supported with rg."
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
                } else if input.head_limit > 0 {
                    let lines: Vec<&str> = text.lines().collect();
                    if lines.len() > input.head_limit {
                        let truncated: String = lines[..input.head_limit].join("\n");
                        let omitted = lines.len() - input.head_limit;
                        ToolResult::success(format!(
                            "{truncated}\n\n... ({omitted} more lines, use head_limit to see more)"
                        ))
                    } else {
                        ToolResult::success(text)
                    }
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

    if input.multiline {
        cmd.arg("-U").arg("--multiline-dotall");
    }

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
            // Context lines: context is shorthand for -A + -B.
            if let Some(c) = input.context {
                cmd.arg("-C").arg(c.to_string());
            } else {
                if let Some(a) = input.after_context {
                    cmd.arg("-A").arg(a.to_string());
                }
                if let Some(b) = input.before_context {
                    cmd.arg("-B").arg(b.to_string());
                }
            }
        }
    }

    if input.case_insensitive {
        cmd.arg("-i");
    }
    if let Some(ref g) = input.glob {
        cmd.arg("--glob").arg(g);
    }
    if let Some(ref t) = input.file_type {
        cmd.arg("--type").arg(t);
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
            if let Some(c) = input.context {
                cmd.arg("-C").arg(c.to_string());
            } else {
                if let Some(a) = input.after_context {
                    cmd.arg("-A").arg(a.to_string());
                }
                if let Some(b) = input.before_context {
                    cmd.arg("-B").arg(b.to_string());
                }
            }
        }
    }

    if input.case_insensitive {
        cmd.arg("-i");
    }
    if let Some(ref g) = input.glob {
        cmd.arg("--include").arg(g);
    }
    // grep doesn't have --type; approximate with --include patterns.
    // Map common rg type names to file extensions.
    if let Some(ref t) = input.file_type {
        let extensions = match t.as_str() {
            "rust" => vec!["rs"],
            "js" | "javascript" => vec!["js", "mjs", "cjs"],
            "ts" | "typescript" => vec!["ts", "mts", "cts"],
            "py" | "python" => vec!["py"],
            "go" => vec!["go"],
            "java" => vec!["java"],
            "css" => vec!["css"],
            "html" => vec!["html", "htm"],
            "json" => vec!["json"],
            "md" | "markdown" => vec!["md"],
            "yaml" | "yml" => vec!["yaml", "yml"],
            "toml" => vec!["toml"],
            "xml" => vec!["xml"],
            "sh" | "shell" => vec!["sh", "bash", "zsh"],
            "c" => vec!["c", "h"],
            "cpp" => vec!["cpp", "cc", "cxx", "hpp", "hxx", "h"],
            _ => vec![t.as_str()],
        };
        for ext in extensions {
            cmd.arg("--include").arg(format!("*.{ext}"));
        }
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
    async fn head_limit_truncates() {
        let dir = tempfile::tempdir().unwrap();
        // Create a file with many matching lines.
        let content: String = (0..100).map(|i| format!("match line {i}\n")).collect();
        fs::write(dir.path().join("big.txt"), &content).unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "match",
                    "path": dir.path().join("big.txt").to_str().unwrap(),
                    "output_mode": "content",
                    "head_limit": 5
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        // Should have exactly 5 result lines plus the truncation marker.
        assert!(
            text.contains("more lines"),
            "should be truncated, got: {text}"
        );
        let lines: Vec<&str> = text.lines().collect();
        // 5 content lines + 1 blank + 1 marker = 7
        assert!(
            lines.len() < 10,
            "should be limited, got {} lines",
            lines.len()
        );
    }

    #[tokio::test]
    async fn head_limit_zero_unlimited() {
        let dir = tempfile::tempdir().unwrap();
        let content: String = (0..10).map(|i| format!("line {i}\n")).collect();
        fs::write(dir.path().join("test.txt"), &content).unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "line",
                    "path": dir.path().join("test.txt").to_str().unwrap(),
                    "output_mode": "content",
                    "head_limit": 0
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(!text.contains("more lines"), "should not be truncated");
    }

    #[tokio::test]
    async fn context_lines() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("test.txt"),
            "line1\nline2\nTARGET\nline4\nline5",
        )
        .unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "TARGET",
                    "path": dir.path().join("test.txt").to_str().unwrap(),
                    "output_mode": "content",
                    "context": 1
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(
            text.contains("line2"),
            "should include before context: {text}"
        );
        assert!(text.contains("TARGET"));
        assert!(
            text.contains("line4"),
            "should include after context: {text}"
        );
    }

    #[tokio::test]
    async fn after_context() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.txt"), "before\nMATCH\nafter1\nafter2").unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "MATCH",
                    "path": dir.path().join("test.txt").to_str().unwrap(),
                    "output_mode": "content",
                    "-A": 2
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("MATCH"));
        assert!(
            text.contains("after1"),
            "should include after context: {text}"
        );
        assert!(
            text.contains("after2"),
            "should include after context: {text}"
        );
    }

    #[tokio::test]
    async fn type_filter() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("code.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("data.json"), "fn main() {}").unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "fn main",
                    "path": dir.path().to_str().unwrap(),
                    "type": "rust"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("code.rs"), "should match .rs file: {text}");
        assert!(
            !text.contains("data.json"),
            "should not match .json file: {text}"
        );
    }

    #[tokio::test]
    async fn multiline_search() {
        // multiline requires rg; skip if not available.
        let has_rg = std::process::Command::new("rg")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if !has_rg {
            eprintln!("skipping multiline_search: rg not installed");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("test.txt"),
            "struct Foo {\n    field: i32,\n}",
        )
        .unwrap();

        let tool = GrepTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "pattern": "struct Foo \\{[\\s\\S]*?field",
                    "path": dir.path().join("test.txt").to_str().unwrap(),
                    "output_mode": "content",
                    "multiline": true
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(
            text.contains("struct Foo"),
            "should match across lines: {text}"
        );
        assert!(text.contains("field"), "should include field: {text}");
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
