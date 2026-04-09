use serde::{Deserialize, Serialize};

pub fn new_uuid() -> String {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).expect("failed to generate random bytes");
    buf[6] = (buf[6] & 0x0f) | 0x40;
    buf[8] = (buf[8] & 0x3f) | 0x80;
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]),
        u16::from_be_bytes([buf[4], buf[5]]),
        u16::from_be_bytes([buf[6], buf[7]]),
        u16::from_be_bytes([buf[8], buf[9]]),
        u64::from_be_bytes([0, 0, buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]]),
    )
}

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
    /// Server-side tool use (e.g. web_search). Executed by the provider, not locally.
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Result from a server-side tool (e.g. web_search).
    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
    Thinking {
        thinking: String,
        /// Signature for verifying thinking integrity when sent back to the API.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Redacted thinking block — opaque data that must be passed back verbatim.
    RedactedThinking {
        data: String,
    },
}

/// Content within a tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image { source: ImageSource },
}

/// Base64-encoded image source for tool results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
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

    pub fn image(media_type: impl Into<String>, base64_data: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: media_type.into(),
                    data: base64_data.into(),
                },
            }],
            is_error: false,
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
            content: vec![ToolResultContent::Text { text: "ok".into() }],
            is_error: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert!(json.get("is_error").is_none());
    }

    #[test]
    fn roundtrip_message() {
        let msg = Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "hi".into() }],
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

    #[test]
    fn serialize_server_tool_use() {
        let block = ContentBlock::ServerToolUse {
            id: "stu_01".into(),
            name: "web_search".into(),
            input: json!({"query": "rust lang"}),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "server_tool_use");
        assert_eq!(json["name"], "web_search");
    }

    #[test]
    fn serialize_web_search_tool_result() {
        let block = ContentBlock::WebSearchToolResult {
            tool_use_id: "stu_01".into(),
            content: json!([{"title": "Rust", "url": "https://rust-lang.org"}]),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "web_search_tool_result");
        assert_eq!(json["tool_use_id"], "stu_01");
    }

    #[test]
    fn serialize_thinking_with_signature() {
        let block = ContentBlock::Thinking {
            thinking: "let me think".into(),
            signature: Some("sig123".into()),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "thinking");
        assert_eq!(json["thinking"], "let me think");
        assert_eq!(json["signature"], "sig123");
    }

    #[test]
    fn serialize_thinking_without_signature() {
        let block = ContentBlock::Thinking {
            thinking: "hmm".into(),
            signature: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "thinking");
        assert_eq!(json["thinking"], "hmm");
        assert!(json.get("signature").is_none());
    }

    #[test]
    fn serialize_redacted_thinking() {
        let block = ContentBlock::RedactedThinking {
            data: "opaque_data".into(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "redacted_thinking");
        assert_eq!(json["data"], "opaque_data");
    }

    #[test]
    fn roundtrip_redacted_thinking() {
        let block = ContentBlock::RedactedThinking {
            data: "opaque".into(),
        };
        let serialized = serde_json::to_string(&block).unwrap();
        let deserialized: ContentBlock = serde_json::from_str(&serialized).unwrap();
        assert!(
            matches!(deserialized, ContentBlock::RedactedThinking { data } if data == "opaque")
        );
    }

    #[test]
    fn roundtrip_thinking_with_signature() {
        let block = ContentBlock::Thinking {
            thinking: "deep thought".into(),
            signature: Some("sig_abc".into()),
        };
        let serialized = serde_json::to_string(&block).unwrap();
        let deserialized: ContentBlock = serde_json::from_str(&serialized).unwrap();
        assert!(matches!(
            deserialized,
            ContentBlock::Thinking {
                thinking,
                signature: Some(sig)
            } if thinking == "deep thought" && sig == "sig_abc"
        ));
    }

    #[test]
    fn serialize_stream_event_system() {
        let event = StreamEvent::System {
            subtype: "init".into(),
            data: json!({"tools": ["bash"], "model": "test"}),
            session_id: "sid".into(),
            uuid: "uid".into(),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "system");
        assert_eq!(json["subtype"], "init");
        assert_eq!(json["tools"], json!(["bash"]));
        assert_eq!(json["session_id"], "sid");
    }

    #[test]
    fn serialize_stream_event_assistant() {
        let event = StreamEvent::Assistant {
            message: json!({
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "id": "msg_01",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
            }),
            parent_tool_use_id: json!(null),
            session_id: "sid".into(),
            uuid: "uid".into(),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "assistant");
        assert_eq!(json["message"]["role"], "assistant");
        assert_eq!(json["message"]["model"], "test-model");
        assert_eq!(json["message"]["id"], "msg_01");
        assert_eq!(json["message"]["stop_reason"], "end_turn");
        assert!(json["parent_tool_use_id"].is_null());
    }

    #[test]
    fn serialize_stream_event_result() {
        let event = StreamEvent::Result {
            subtype: "success".into(),
            duration_ms: 100,
            is_error: false,
            num_turns: 1,
            result: Some("done".into()),
            stop_reason: Some("end_turn".into()),
            session_id: "sid".into(),
            uuid: "uid".into(),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "result");
        assert_eq!(json["subtype"], "success");
        assert_eq!(json["is_error"], false);
        assert_eq!(json["result"], "done");
        assert_eq!(json["stop_reason"], "end_turn");
    }

    #[test]
    fn serialize_stream_event_result_error_skips_none() {
        let event = StreamEvent::Result {
            subtype: "error_during_execution".into(),
            duration_ms: 50,
            is_error: true,
            num_turns: 1,
            result: None,
            stop_reason: None,
            session_id: "sid".into(),
            uuid: "uid".into(),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "result");
        assert!(json.get("result").is_none());
        assert!(json.get("stop_reason").is_none());
    }

    #[test]
    fn serialize_stream_event_user() {
        let event = StreamEvent::User {
            message: Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "hello".into(),
                }],
            },
            parent_tool_use_id: json!(null),
            session_id: "sid".into(),
            uuid: "uid".into(),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "user");
        assert_eq!(json["message"]["role"], "user");
        assert_eq!(json["message"]["content"][0]["text"], "hello");
    }

    #[test]
    fn serialize_stream_event_raw() {
        let event = StreamEvent::RawEvent {
            event: json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}),
            parent_tool_use_id: json!(null),
            session_id: "sid".into(),
            uuid: "uid".into(),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "stream_event");
        assert_eq!(json["event"]["delta"]["text"], "hi");
    }
}

/// Events streamed from the agent loop (Claude Code stream-json compatible).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    /// Session initialization.
    #[serde(rename = "system")]
    System {
        subtype: String,
        #[serde(flatten)]
        data: serde_json::Value,
        session_id: String,
        uuid: String,
    },
    /// Complete assistant message (full API response format).
    #[serde(rename = "assistant")]
    Assistant {
        message: serde_json::Value,
        parent_tool_use_id: serde_json::Value,
        session_id: String,
        uuid: String,
    },
    /// User message (prompt or tool results).
    #[serde(rename = "user")]
    User {
        message: Message,
        parent_tool_use_id: serde_json::Value,
        session_id: String,
        uuid: String,
    },
    /// Raw API stream event for real-time streaming.
    #[serde(rename = "stream_event")]
    RawEvent {
        event: serde_json::Value,
        parent_tool_use_id: serde_json::Value,
        session_id: String,
        uuid: String,
    },
    /// Final result.
    #[serde(rename = "result")]
    Result {
        subtype: String,
        duration_ms: u64,
        is_error: bool,
        num_turns: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
        session_id: String,
        uuid: String,
    },
}
