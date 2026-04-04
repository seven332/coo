use serde::{Deserialize, Serialize};

/// A role in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

/// A content block within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    Thinking {
        thinking: String,
    },
}

/// Content within a tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
}

/// The result returned by a tool execution.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: Vec<ToolResultContent>,
    pub is_error: bool,
}

impl ToolResult {
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text { text: text.into() }],
            is_error: false,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text { text: text.into() }],
            is_error: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serialize_text_block() {
        let block = ContentBlock::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hello");
    }

    #[test]
    fn serialize_tool_use_block() {
        let block = ContentBlock::ToolUse {
            id: "toolu_01".into(),
            name: "bash".into(),
            input: json!({"command": "ls"}),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["name"], "bash");
        assert_eq!(json["input"]["command"], "ls");
    }

    #[test]
    fn tool_result_block_skips_none_error() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "toolu_01".into(),
            content: vec![ToolResultContent::Text {
                text: "ok".into(),
            }],
            is_error: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert!(json.get("is_error").is_none());
    }

    #[test]
    fn roundtrip_message() {
        let msg = Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "hi".into(),
            }],
        };
        let serialized = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.role, Role::User);
        assert!(matches!(&deserialized.content[0], ContentBlock::Text { text } if text == "hi"));
    }

    #[test]
    fn deserialize_tool_use_from_api() {
        let json = json!({
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "read",
            "input": {"file_path": "/tmp/test.txt"}
        });
        let block: ContentBlock = serde_json::from_value(json).unwrap();
        assert!(matches!(block, ContentBlock::ToolUse { name, .. } if name == "read"));
    }

    #[test]
    fn tool_result_helpers() {
        let ok = ToolResult::success("done");
        assert!(!ok.is_error);

        let err = ToolResult::error("fail");
        assert!(err.is_error);
    }
}

/// Events streamed from the agent loop.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Assistant is producing text.
    Text { text: String },
    /// Assistant is thinking.
    Thinking { thinking: String },
    /// Assistant wants to use a tool.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A tool has produced a result.
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    /// The agent loop has finished.
    Done {
        /// Why the loop ended: "end_turn", "max_tokens", "error"
        reason: String,
    },
    /// An error occurred.
    Error { message: String },
}
