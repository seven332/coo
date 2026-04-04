use anyhow::{Context as _, bail};
use async_trait::async_trait;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::{Chunk, Provider, Request, StopReason};
use crate::message::ContentBlock;

pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
        }
    }
}

// --- API types ---

#[derive(Serialize)]
struct ApiRequest<'a> {
    model: &'a str,
    system: &'a str,
    messages: &'a [crate::message::Message],
    tools: &'a [super::ToolDefinition],
    max_tokens: u32,
    stream: bool,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum SseEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: serde_json::Value },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockInfo,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: Delta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaBody,
        #[serde(default)]
        usage: serde_json::Value,
    },
    #[serde(rename = "message_stop")]
    MessageStop {},
    #[serde(rename = "ping")]
    Ping {},
    #[serde(rename = "error")]
    Error { error: ApiError },
}

#[derive(Deserialize, Debug)]
struct ApiError {
    message: String,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type")]
enum ContentBlockInfo {
    #[serde(rename = "text")]
    Text {
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: String,
    },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum Delta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature {
        #[allow(dead_code)]
        signature: String,
    },
}

#[derive(Deserialize, Debug)]
struct MessageDeltaBody {
    #[serde(default)]
    stop_reason: Option<String>,
}

// --- Accumulator for building content blocks ---

struct BlockAccumulator {
    info: ContentBlockInfo,
    json_buf: String,
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = ApiRequest {
            model: &req.model,
            system: &req.system,
            messages: &req.messages,
            tools: &req.tools,
            max_tokens: req.max_tokens,
            stream: true,
        };

        let http_req = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body);

        let mut es = EventSource::new(http_req)
            .context("Failed to create event source")?;

        let mut blocks: Vec<BlockAccumulator> = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        use futures::StreamExt as _;
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(msg)) => {
                    let sse: SseEvent = match serde_json::from_str(&msg.data) {
                        Ok(e) => e,
                        Err(e) => {
                            let _ = tx.send(Chunk::Error(format!("Parse error: {e}"))).await;
                            break;
                        }
                    };

                    match sse {
                        SseEvent::Ping {} | SseEvent::MessageStart { .. } => {}

                        SseEvent::ContentBlockStart {
                            index,
                            content_block,
                        } => {
                            // Ensure blocks vec is large enough.
                            while blocks.len() <= index {
                                blocks.push(BlockAccumulator {
                                    info: ContentBlockInfo::Text {
                                        text: String::new(),
                                    },
                                    json_buf: String::new(),
                                });
                            }
                            blocks[index] = BlockAccumulator {
                                info: content_block,
                                json_buf: String::new(),
                            };
                        }

                        SseEvent::ContentBlockDelta { index, delta } => {
                            if let Some(block) = blocks.get_mut(index) {
                                match delta {
                                    Delta::Text { text } => {
                                        let _ = tx.send(Chunk::Text(text)).await;
                                    }
                                    Delta::InputJson { partial_json } => {
                                        block.json_buf.push_str(&partial_json);
                                    }
                                    Delta::Thinking { thinking } => {
                                        let _ = tx.send(Chunk::Thinking(thinking)).await;
                                    }
                                    Delta::Signature { .. } => {}
                                }
                            }
                        }

                        SseEvent::ContentBlockStop { index } => {
                            if let Some(block) = blocks.get_mut(index) {
                                if let ContentBlockInfo::ToolUse { ref id, ref name } = block.info {
                                    let input: serde_json::Value =
                                        serde_json::from_str(&block.json_buf).unwrap_or_default();
                                    let _ = tx
                                        .send(Chunk::ToolUse {
                                            id: id.clone(),
                                            name: name.clone(),
                                            input,
                                        })
                                        .await;
                                }
                            }
                        }

                        SseEvent::MessageDelta { delta, .. } => {
                            if let Some(reason) = delta.stop_reason {
                                stop_reason = match reason.as_str() {
                                    "end_turn" => StopReason::EndTurn,
                                    "tool_use" => StopReason::ToolUse,
                                    "max_tokens" => StopReason::MaxTokens,
                                    _ => StopReason::EndTurn,
                                };
                            }
                        }

                        SseEvent::MessageStop {} => {
                            break;
                        }

                        SseEvent::Error { error } => {
                            let _ = tx.send(Chunk::Error(error.message)).await;
                            break;
                        }
                    }
                }
                Err(reqwest_eventsource::Error::StreamEnded) => break,
                Err(e) => {
                    bail!("SSE error: {e}");
                }
            }
        }

        // Build final content blocks.
        let content = blocks
            .into_iter()
            .map(|b| match b.info {
                ContentBlockInfo::Text { text } => ContentBlock::Text { text },
                ContentBlockInfo::ToolUse { id, name } => {
                    let input = serde_json::from_str(&b.json_buf).unwrap_or_default();
                    ContentBlock::ToolUse { id, name, input }
                }
                ContentBlockInfo::Thinking { thinking } => ContentBlock::Thinking { thinking },
            })
            .collect();

        let _ = tx.send(Chunk::Done {
            stop_reason,
            content,
        }).await;

        Ok(())
    }
}
