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
    #[serde(default)]
    pages: Option<String>,
}

/// Parse a pages specification like "1-5", "3", "10-20" into a list of 0-based page indices.
/// Returns an error message if the format is invalid.
fn parse_pages(spec: &str, total_pages: usize) -> Result<Vec<usize>, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err("Empty pages specification".into());
    }

    let mut indices = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((start_s, end_s)) = part.split_once('-') {
            let start: usize = start_s
                .trim()
                .parse()
                .map_err(|_| format!("Invalid page number: {start_s}"))?;
            let end: usize = end_s
                .trim()
                .parse()
                .map_err(|_| format!("Invalid page number: {end_s}"))?;
            if start == 0 || end == 0 {
                return Err("Page numbers start at 1".into());
            }
            if start > end {
                return Err(format!("Invalid range: {start}-{end}"));
            }
            if start > total_pages {
                return Err(format!(
                    "Page {start} out of range (document has {total_pages} pages)"
                ));
            }
            let end = end.min(total_pages);
            for i in start..=end {
                indices.push(i - 1); // Convert to 0-based
            }
        } else {
            let page: usize = part
                .parse()
                .map_err(|_| format!("Invalid page number: {part}"))?;
            if page == 0 {
                return Err("Page numbers start at 1".into());
            }
            if page > total_pages {
                return Err(format!(
                    "Page {page} out of range (document has {total_pages} pages)"
                ));
            }
            indices.push(page - 1);
        }
    }

    if indices.len() > 20 {
        return Err(format!(
            "Too many pages requested ({}, max 20)",
            indices.len()
        ));
    }

    Ok(indices)
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
                },
                "pages": {
                    "type": "string",
                    "description": "Page range for PDF files (e.g. \"1-5\", \"3\", \"10-20\"). Only for PDF files. Max 20 pages per request."
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

        // Handle PDF files: extract text from pages.
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
        {
            let bytes = match tokio::fs::read(&path).await {
                Ok(b) => b,
                Err(e) => {
                    return ToolResult::error(format!("Failed to read {}: {e}", path.display()));
                }
            };
            // Reject PDFs larger than 100 MB to prevent excessive memory usage during parsing.
            const MAX_PDF_BYTES: usize = 100 * 1024 * 1024;
            if bytes.len() > MAX_PDF_BYTES {
                return ToolResult::error(format!(
                    "PDF too large ({} bytes, max {})",
                    bytes.len(),
                    MAX_PDF_BYTES
                ));
            }
            // pdf-extract is synchronous; run on blocking thread pool.
            let result = tokio::task::spawn_blocking(move || {
                pdf_extract::extract_text_from_mem_by_pages(&bytes)
            })
            .await;
            let all_pages = match result {
                Ok(Ok(pages)) => pages,
                Ok(Err(e)) => {
                    return ToolResult::error(format!("Failed to parse PDF: {e}"));
                }
                Err(e) => {
                    return ToolResult::error(format!("PDF extraction task failed: {e}"));
                }
            };

            let total = all_pages.len();
            if total == 0 {
                return ToolResult::success("(empty PDF — no pages)");
            }

            let indices = match &input.pages {
                Some(spec) => match parse_pages(spec, total) {
                    Ok(idx) => idx,
                    Err(e) => return ToolResult::error(e),
                },
                None => {
                    if total > 20 {
                        return ToolResult::error(format!(
                            "PDF has {total} pages. Provide a `pages` parameter to select \
                             which pages to read (max 20 per request). Example: \"1-5\""
                        ));
                    }
                    (0..total).collect()
                }
            };

            let mut output = String::new();
            for &idx in &indices {
                let page_num = idx + 1;
                output.push_str(&format!("--- Page {page_num} of {total} ---\n"));
                output.push_str(all_pages[idx].trim());
                output.push('\n');
            }
            return ToolResult::success(output);
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

    // --- parse_pages tests ---

    #[test]
    fn parse_pages_single() {
        let result = parse_pages("3", 10).unwrap();
        assert_eq!(result, vec![2]); // 0-based
    }

    #[test]
    fn parse_pages_range() {
        let result = parse_pages("2-4", 10).unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn parse_pages_comma_separated() {
        let result = parse_pages("1, 3, 5", 10).unwrap();
        assert_eq!(result, vec![0, 2, 4]);
    }

    #[test]
    fn parse_pages_mixed() {
        let result = parse_pages("1-3, 7", 10).unwrap();
        assert_eq!(result, vec![0, 1, 2, 6]);
    }

    #[test]
    fn parse_pages_clamps_end() {
        // Range end exceeds total pages — should clamp.
        let result = parse_pages("8-15", 10).unwrap();
        assert_eq!(result, vec![7, 8, 9]);
    }

    #[test]
    fn parse_pages_zero_error() {
        assert!(parse_pages("0", 10).is_err());
    }

    #[test]
    fn parse_pages_out_of_range() {
        assert!(parse_pages("11", 10).is_err());
    }

    #[test]
    fn parse_pages_reversed_range() {
        assert!(parse_pages("5-3", 10).is_err());
    }

    #[test]
    fn parse_pages_too_many() {
        // 1-21 = 21 pages, exceeds max 20.
        assert!(parse_pages("1-21", 30).is_err());
    }

    #[test]
    fn parse_pages_empty() {
        assert!(parse_pages("", 10).is_err());
    }

    // --- PDF read tests ---

    /// Create a minimal valid PDF with two pages using lopdf (transitive dep of pdf-extract).
    fn make_test_pdf() -> Vec<u8> {
        use pdf_extract::{Document, dictionary};

        let mut doc = Document::with_version("1.4");
        let font_id = doc.add_object(pdf_extract::dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });

        let pages_id = doc.new_object_id();
        let mut page_ids = vec![];

        for text in &["Hello", "World"] {
            let content = format!("BT /F1 12 Tf 100 700 Td ({text}) Tj ET");
            let content_id = doc.add_object(pdf_extract::Stream::new(
                Default::default(),
                content.into_bytes(),
            ));

            let page_id = doc.add_object(pdf_extract::dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                "Contents" => content_id,
                "Resources" => pdf_extract::dictionary! {
                    "Font" => pdf_extract::dictionary! {
                        "F1" => font_id,
                    },
                },
            });
            page_ids.push(page_id);
        }

        let pages = pdf_extract::dictionary! {
            "Type" => "Pages",
            "Kids" => page_ids.into_iter().map(pdf_extract::Object::Reference).collect::<Vec<_>>(),
            "Count" => 2,
        };
        doc.objects
            .insert(pages_id, pdf_extract::Object::Dictionary(pages));

        let catalog_id = doc.add_object(pdf_extract::dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);

        let mut buf = Vec::new();
        doc.save_to(&mut buf).unwrap();
        buf
    }

    #[tokio::test]
    async fn read_pdf_all_pages() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pdf");
        std::fs::write(&path, make_test_pdf()).unwrap();

        let tool = ReadTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(json!({"file_path": path.to_str().unwrap()}), &ctx)
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(text.contains("Page 1"));
        assert!(text.contains("Page 2"));
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[tokio::test]
    async fn read_pdf_specific_page() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pdf");
        std::fs::write(&path, make_test_pdf()).unwrap();

        let tool = ReadTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({"file_path": path.to_str().unwrap(), "pages": "2"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = text_of(&result);
        assert!(!text.contains("Page 1"));
        assert!(text.contains("Page 2"));
        assert!(text.contains("World"));
    }

    #[tokio::test]
    async fn read_pdf_invalid_page() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pdf");
        std::fs::write(&path, make_test_pdf()).unwrap();

        let tool = ReadTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(
                json!({"file_path": path.to_str().unwrap(), "pages": "5"}),
                &ctx,
            )
            .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn read_invalid_pdf() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.pdf");
        std::fs::write(&path, b"not a pdf").unwrap();

        let tool = ReadTool;
        let ctx = crate::tools::dummy_context();
        let result = tool
            .call(json!({"file_path": path.to_str().unwrap()}), &ctx)
            .await;
        assert!(result.is_error);
    }
}
