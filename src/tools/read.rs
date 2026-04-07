use async_trait::async_trait;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::json;

use crate::message::ToolResult;

use super::Tool;

pub struct ReadTool;

/// Detect media type from file extension. Returns None for non-image files.
fn image_media_type(path: &std::path::Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

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

    async fn call(&self, input: serde_json::Value, context: &super::ToolContext) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        let path = context.resolve_path(&input.file_path);

        // Handle image files: return as base64-encoded image content block.
        if let Some(media_type) = image_media_type(&path) {
            let bytes = match tokio::fs::read(&path).await {
                Ok(b) => b,
                Err(e) => {
                    return ToolResult::error(format!("Failed to read {}: {e}", path.display()));
                }
            };
            // Reject images larger than 10 MB to avoid excessive token usage.
            const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
            if bytes.len() > MAX_IMAGE_BYTES {
                return ToolResult::error(format!(
                    "Image too large ({} bytes, max {})",
                    bytes.len(),
                    MAX_IMAGE_BYTES
                ));
            }
            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
            return ToolResult::image(media_type, b64);
        }

        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to read {}: {e}", path.display())),
        };

        let lines: Vec<&str> = content.lines().collect();
        let offset = input.offset.unwrap_or(0);
        let limit = input.limit.unwrap_or(2000);
        let end = (offset + limit).min(lines.len());

        if lines.is_empty() {
            return ToolResult::success("(empty file)");
        }

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
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn read_file() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "line1").unwrap();
        writeln!(tmp, "line2").unwrap();
        writeln!(tmp, "line3").unwrap();

        let tool = ReadTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(json!({"file_path": tmp.path().to_str().unwrap()}), &ctx)
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
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({
                    "file_path": tmp.path().to_str().unwrap(),
                    "offset": 2,
                    "limit": 3
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("3\tline3"));
        assert!(text.contains("5\tline5"));
        assert!(!text.contains("6\tline6"));
    }

    #[tokio::test]
    async fn read_empty_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();

        let tool = ReadTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(json!({"file_path": tmp.path().to_str().unwrap()}), &ctx)
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("empty"));
    }

    #[tokio::test]
    async fn read_nonexistent() {
        let tool = ReadTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(json!({"file_path": "/tmp/nonexistent_coo_test_file"}), &ctx)
            .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn read_binary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("binary.bin");
        std::fs::write(&path, b"\x00\xff\xfe").unwrap();

        let tool = ReadTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(json!({"file_path": path.to_str().unwrap()}), &ctx)
            .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn read_png_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.png");
        // Minimal valid PNG (1x1 pixel).
        let png_data = b"\x89PNG\r\n\x1a\n";
        std::fs::write(&path, png_data).unwrap();

        let tool = ReadTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(json!({"file_path": path.to_str().unwrap()}), &ctx)
            .await;
        assert!(!result.is_error);
        match &result.content[0] {
            crate::message::ToolResultContent::Image { source } => {
                assert_eq!(source.media_type, "image/png");
                assert_eq!(source.source_type, "base64");
                assert!(!source.data.is_empty());
            }
            _ => panic!("expected image content"),
        }
    }

    #[tokio::test]
    async fn read_jpg_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jpg");
        std::fs::write(&path, b"\xff\xd8\xff").unwrap();

        let tool = ReadTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(json!({"file_path": path.to_str().unwrap()}), &ctx)
            .await;
        assert!(!result.is_error);
        match &result.content[0] {
            crate::message::ToolResultContent::Image { source } => {
                assert_eq!(source.media_type, "image/jpeg");
            }
            _ => panic!("expected image content"),
        }
    }

    #[tokio::test]
    async fn read_relative_with_cwd() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hello.txt"), "content here\n").unwrap();

        let tool = ReadTool;
        let mut ctx = crate::tools::dummy_context();
        ctx.cwd = Some(dir.path().to_str().unwrap().to_string());
        let result = tool.call(json!({"file_path": "hello.txt"}), &ctx).await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("content here"));
    }
}
