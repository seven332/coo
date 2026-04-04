use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::message::ToolResult;

use super::Tool;

pub struct WebFetchTool {
    client: reqwest::Client,
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .redirect(reqwest::redirect::Policy::limited(10))
                .build()
                .expect("failed to build HTTP client"),
        }
    }
}

#[derive(Deserialize)]
struct Input {
    url: String,
    #[serde(default = "default_max_length")]
    max_length: usize,
}

fn default_max_length() -> usize {
    100_000
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch content from a URL. Returns the page content as markdown. \
         HTML is converted to readable markdown format."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum content length in characters (default: 100000)"
                }
            },
            "required": ["url"]
        })
    }

    async fn call(&self, input: serde_json::Value, _context: &super::ToolContext) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        if input.url.len() > 2000 {
            return ToolResult::error("URL exceeds 2000 character limit");
        }

        // Upgrade http to https.
        let url = if input.url.starts_with("http://") {
            input.url.replacen("http://", "https://", 1)
        } else {
            input.url
        };

        let response = match self
            .client
            .get(&url)
            .header("Accept", "text/html, text/plain, */*")
            .header("User-Agent", "you-mind/0.1")
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return ToolResult::error(format!("Fetch failed: {e}")),
        };

        let status = response.status();
        if !status.is_success() {
            return ToolResult::error(format!("HTTP {status}"));
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let bytes = match response.bytes().await {
            Ok(b) => b,
            Err(e) => return ToolResult::error(format!("Failed to read response: {e}")),
        };

        let text = String::from_utf8_lossy(&bytes);

        let content = if content_type.contains("text/html") {
            let converter = htmd::HtmlToMarkdown::builder()
                .skip_tags(vec!["script", "style"])
                .build();
            converter
                .convert(&text)
                .unwrap_or_else(|_| text.into_owned())
        } else {
            text.into_owned()
        };

        let truncated = if content.len() > input.max_length {
            // Find a valid char boundary at or before max_length.
            let end = content.floor_char_boundary(input.max_length);
            format!(
                "{}\n\n[Content truncated at {} bytes]",
                &content[..end],
                end
            )
        } else {
            content
        };

        ToolResult::success(truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_to_markdown() {
        let html = "<html><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>";
        let md = htmd::convert(html).unwrap();
        assert!(md.contains("# Title"));
        assert!(md.contains("**world**"));
    }

    #[test]
    fn html_links_preserved() {
        let html = r#"<p>Visit <a href="https://example.com">Example</a></p>"#;
        let md = htmd::convert(html).unwrap();
        assert!(md.contains("[Example](https://example.com)"));
    }

    #[test]
    fn html_lists() {
        let html = "<ul><li>one</li><li>two</li></ul>";
        let md = htmd::convert(html).unwrap();
        assert!(md.contains("one"));
        assert!(md.contains("two"));
    }

    #[test]
    fn html_script_style_stripped() {
        let html = "<p>Before</p><script>var x=1;</script><style>.a{}</style><p>After</p>";
        let converter = htmd::HtmlToMarkdown::builder()
            .skip_tags(vec!["script", "style"])
            .build();
        let md = converter.convert(html).unwrap();
        assert!(md.contains("Before"));
        assert!(md.contains("After"));
        assert!(!md.contains("var x"));
        assert!(!md.contains(".a{}"));
    }

    #[tokio::test]
    async fn url_too_long() {
        let tool = WebFetchTool::new();
        let ctx = crate::tools::dummy_context();
        let long_url = format!("https://example.com/{}", "a".repeat(2000));
        let result = tool.call(json!({"url": long_url}), &ctx).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn invalid_input() {
        let tool = WebFetchTool::new();
        let ctx = crate::tools::dummy_context();
        let result = tool.call(json!({}), &ctx).await;
        assert!(result.is_error);
    }
}
