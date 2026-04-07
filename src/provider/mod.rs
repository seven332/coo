mod anthropic;
mod coo;

pub use anthropic::AnthropicProvider;
pub use coo::CooProvider;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::message::{ContentBlock, Message};

/// Tool definition sent to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// A server-side tool managed by the provider (e.g. web_search).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerTool {
    /// Anthropic web search server tool.
    #[serde(rename = "web_search_20250305")]
    WebSearch {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_uses: Option<u32>,
    },
}

/// Request to the LLM.
#[derive(Debug, Clone)]
pub struct Request {
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub server_tools: Vec<ServerTool>,
    pub max_tokens: u32,
    /// Message index at which to insert a cache breakpoint.
    /// The last content block of messages[cache_breakpoint] will get
    /// `cache_control: {"type": "ephemeral"}` to enable prompt cache reuse
    /// across forked sub-agents sharing the same message prefix.
    pub cache_breakpoint: Option<usize>,
}

/// Metadata from the API response (model, message id, usage).
#[derive(Debug, Clone, Default)]
pub struct ResponseMeta {
    pub model: String,
    pub message_id: String,
    pub usage: serde_json::Value,
}

/// Streamed chunk from the LLM.
#[derive(Debug, Clone)]
pub enum Chunk {
    /// Incremental text output.
    Text(String),
    /// Thinking/reasoning text.
    Thinking(String),
    /// A complete tool use block.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// API response metadata (sent before Done).
    Meta(ResponseMeta),
    /// The response is done.
    Done {
        stop_reason: StopReason,
        content: Vec<ContentBlock>,
    },
    /// An error occurred.
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

/// Trait for LLM providers.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Send a request and stream chunks back via the channel.
    async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()>;
}
