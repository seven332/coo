mod anthropic;
mod meow;

pub use anthropic::AnthropicProvider;
pub use meow::MeowProvider;

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

/// Request to the LLM.
#[derive(Debug, Clone)]
pub struct Request {
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
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
