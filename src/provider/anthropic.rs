use anyhow::{Context as _, bail};
use async_trait::async_trait;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::{Chunk, Provider, Request, ResponseMeta, StopReason};
use crate::message::{ContentBlock, new_uuid as uuid};

pub struct AnthropicProvider {
    client: reqwest::Client,
    token: String,
    is_oauth: bool,
    base_url: String,
    session_id: String,
}

impl AnthropicProvider {
    pub fn new(token: String, base_url: Option<String>) -> Self {
        let is_oauth = token.starts_with("sk-ant-oat");
        Self {
            client: reqwest::Client::new(),
            token,
            is_oauth,
            base_url: base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
            session_id: uuid(),
        }
    }
}

// --- API types ---

#[derive(Serialize)]
struct ApiRequest<'a> {
    model: &'a str,
    system: Vec<SystemBlock<'a>>,
    messages: serde_json::Value,
    tools: Vec<serde_json::Value>,
    max_tokens: u32,
    stream: bool,
    metadata: ApiMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_management: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct SystemBlock<'a> {
    #[serde(rename = "type")]
    block_type: &'a str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    cache_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    scope: Option<String>,
}

#[derive(Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
}

#[derive(Serialize)]
struct ApiMetadata {
    user_id: String,
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
    #[serde(rename = "server_tool_use")]
    ServerToolUse { id: String, name: String },
    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
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
    Signature { signature: String },
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
        let url = format!("{}/v1/messages?beta=true", self.base_url);

        // Merge regular tools and server tools into a single JSON array.
        let mut tools: Vec<serde_json::Value> = req
            .tools
            .iter()
            .filter_map(|t| serde_json::to_value(t).ok())
            .collect();
        for st in &req.server_tools {
            if let Ok(v) = serde_json::to_value(st) {
                tools.push(v);
            }
        }

        // Build system blocks with billing header prepended.
        let system = vec![
            SystemBlock {
                block_type: "text",
                text: "x-anthropic-billing-header: cc_version=2.1.92.f1a; cc_entrypoint=sdk-cli; cch=df561;".to_string(),
                cache_control: None,
            },
            SystemBlock {
                block_type: "text",
                text: req.system.clone(),
                cache_control: Some(CacheControl {
                    cache_type: "ephemeral".to_string(),
                    scope: Some("global".to_string()),
                }),
            },
        ];

        let supports_thinking = !req.model.contains("haiku");

        // Serialize messages and inject cache_control at the breakpoint.
        let messages = {
            let mut msgs = serde_json::to_value(&req.messages).unwrap_or_default();
            if let Some(bp) = req.cache_breakpoint
                && let Some(msg) = msgs.as_array_mut().and_then(|a| a.get_mut(bp))
                && let Some(content) = msg.get_mut("content").and_then(|c| c.as_array_mut())
                && let Some(last_block) = content.last_mut()
            {
                last_block["cache_control"] = serde_json::json!({"type": "ephemeral"});
            }
            msgs
        };

        let body = ApiRequest {
            model: &req.model,
            system,
            messages,
            tools,
            max_tokens: req.max_tokens,
            metadata: ApiMetadata {
                user_id: serde_json::json!({
                    "device_id": self.session_id,
                    "account_uuid": "",
                    "session_id": self.session_id,
                })
                .to_string(),
            },
            stream: true,
            thinking: if supports_thinking {
                Some(ThinkingConfig {
                    thinking_type: "adaptive".to_string(),
                })
            } else {
                None
            },
            context_management: if supports_thinking {
                Some(serde_json::json!({
                    "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
                }))
            } else {
                None
            },
            output_config: if supports_thinking {
                Some(serde_json::json!({"effort": "medium"}))
            } else {
                None
            },
        };

        let mut betas = vec!["claude-code-20250219"];
        if self.is_oauth {
            betas.push("oauth-2025-04-20");
        }
        if supports_thinking {
            betas.extend([
                "interleaved-thinking-2025-05-14",
                "context-management-2025-06-27",
                "effort-2025-11-24",
            ]);
        }
        betas.extend([
            "prompt-caching-scope-2026-01-05",
            "advanced-tool-use-2025-11-20",
        ]);

        let mut http_req = self
            .client
            .post(&url)
            .header("Accept", "application/json")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .header("User-Agent", "claude-cli/2.1.92 (external, sdk-cli)")
            .header("anthropic-dangerous-direct-browser-access", "true")
            .header("x-app", "cli")
            .header("X-Claude-Code-Session-Id", &self.session_id)
            .header("x-client-request-id", uuid())
            .header("X-Stainless-Lang", "js")
            .header("X-Stainless-Runtime", "node")
            .header("X-Stainless-Runtime-Version", "v24.3.0")
            .header("X-Stainless-Arch", "arm64")
            .header("X-Stainless-OS", "Linux")
            .header("X-Stainless-Package-Version", "0.80.0")
            .header("X-Stainless-Retry-Count", "0")
            .header("X-Stainless-Timeout", "600")
            .header("anthropic-beta", betas.join(","));

        if self.is_oauth {
            http_req = http_req.header("Authorization", format!("Bearer {}", self.token));
        } else {
            http_req = http_req.header("x-api-key", &self.token);
        }

        let http_req = http_req.json(&body);

        let mut es = EventSource::new(http_req).context("Failed to create event source")?;

        let mut blocks: Vec<BlockAccumulator> = Vec::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut meta = ResponseMeta {
            model: req.model.clone(),
            ..Default::default()
        };

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
                        SseEvent::Ping {} => {}

                        SseEvent::MessageStart { message } => {
                            if let Some(id) = message.get("id").and_then(|v| v.as_str()) {
                                meta.message_id = id.to_string();
                            }
                            if let Some(model) = message.get("model").and_then(|v| v.as_str()) {
                                meta.model = model.to_string();
                            }
                            if let Some(usage) = message.get("usage") {
                                meta.usage = usage.clone();
                            }
                        }

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
                                    Delta::Text { ref text } => {
                                        if let ContentBlockInfo::Text {
                                            text: ref mut accumulated,
                                        } = block.info
                                        {
                                            accumulated.push_str(text);
                                        }
                                        let _ = tx.send(Chunk::Text(text.clone())).await;
                                    }
                                    Delta::InputJson { partial_json } => {
                                        block.json_buf.push_str(&partial_json);
                                    }
                                    Delta::Thinking { ref thinking } => {
                                        if let ContentBlockInfo::Thinking {
                                            thinking: ref mut accumulated,
                                            ..
                                        } = block.info
                                        {
                                            accumulated.push_str(thinking);
                                        }
                                        let _ = tx.send(Chunk::Thinking(thinking.clone())).await;
                                    }
                                    Delta::Signature { signature } => {
                                        if let ContentBlockInfo::Thinking {
                                            signature: ref mut sig,
                                            ..
                                        } = block.info
                                        {
                                            *sig = Some(
                                                sig.as_deref()
                                                    .map(|s| format!("{s}{signature}"))
                                                    .unwrap_or(signature),
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        SseEvent::ContentBlockStop { index } => {
                            if let Some(block) = blocks.get_mut(index)
                                && let ContentBlockInfo::ToolUse { ref id, ref name } = block.info
                            {
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
                            // ServerToolUse blocks are not sent as Chunk::ToolUse —
                            // the server executes them and returns results inline.
                        }

                        SseEvent::MessageDelta { delta, usage } => {
                            if let Some(reason) = delta.stop_reason {
                                stop_reason = match reason.as_str() {
                                    "end_turn" => StopReason::EndTurn,
                                    "tool_use" => StopReason::ToolUse,
                                    "max_tokens" => StopReason::MaxTokens,
                                    _ => StopReason::EndTurn,
                                };
                            }
                            if let Some(obj) = usage.as_object() {
                                if let Some(base) = meta.usage.as_object_mut() {
                                    for (k, v) in obj {
                                        base.insert(k.clone(), v.clone());
                                    }
                                } else {
                                    meta.usage = usage;
                                }
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
                    let _ = tx.send(Chunk::Error(format!("SSE error: {e}"))).await;
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
                ContentBlockInfo::ServerToolUse { id, name } => {
                    let input = serde_json::from_str(&b.json_buf).unwrap_or_default();
                    ContentBlock::ServerToolUse { id, name, input }
                }
                ContentBlockInfo::WebSearchToolResult {
                    tool_use_id,
                    content,
                } => ContentBlock::WebSearchToolResult {
                    tool_use_id,
                    content,
                },
                ContentBlockInfo::Thinking {
                    thinking,
                    signature,
                } => ContentBlock::Thinking {
                    thinking,
                    signature,
                },
                ContentBlockInfo::RedactedThinking { data } => {
                    ContentBlock::RedactedThinking { data }
                }
            })
            .collect();

        let _ = tx.send(Chunk::Meta(meta)).await;
        let _ = tx
            .send(Chunk::Done {
                stop_reason,
                content,
            })
            .await;

        Ok(())
    }
}
