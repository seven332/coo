use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::message::{ContentBlock, Message, Role, StreamEvent, ToolResultContent};
use crate::provider::{Chunk, Provider, Request, StopReason};
use crate::tools::ToolRegistry;

pub struct Agent {
    provider: Box<dyn Provider>,
    tools: ToolRegistry,
    model: String,
    system: String,
    pub max_tokens: u32,
    pub max_iterations: usize,
}

impl Agent {
    pub fn new(
        provider: Box<dyn Provider>,
        tools: ToolRegistry,
        model: String,
        system: String,
    ) -> Self {
        Self {
            provider,
            tools,
            model,
            system,
            max_tokens: 16384,
            max_iterations: 100,
        }
    }

    /// Run the agent loop. Streams events to `event_tx`.
    /// Takes an initial user prompt and runs until completion.
    pub async fn run(
        &self,
        prompt: String,
        event_tx: mpsc::Sender<StreamEvent>,
    ) -> anyhow::Result<()> {
        let mut messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: prompt }],
        }];

        let tool_defs = self.tools.definitions();

        for iteration in 0..self.max_iterations {
            info!(iteration, "Agent loop iteration");

            // Call LLM.
            let (chunk_tx, mut chunk_rx) = mpsc::channel::<Chunk>(64);

            let req = Request {
                model: self.model.clone(),
                system: self.system.clone(),
                messages: messages.clone(),
                tools: tool_defs.clone(),
                max_tokens: self.max_tokens,
            };

            let provider_handle = {
                let provider = &self.provider;
                let chunk_tx = chunk_tx;
                // We need to spawn this since we want to process chunks as they arrive.
                // But we can't move &self into a spawn. So we'll drive it inline
                // and forward events as they come.
                provider.request(req, chunk_tx)
            };

            // Process chunks concurrently with the provider streaming.
            let event_tx2 = event_tx.clone();
            let (done_tx, done_rx) = tokio::sync::oneshot::channel();

            let chunk_processor = tokio::spawn(async move {
                let mut tool_uses = Vec::new();
                let mut final_content = Vec::new();
                let mut stop = StopReason::EndTurn;

                while let Some(chunk) = chunk_rx.recv().await {
                    match chunk {
                        Chunk::Text(text) => {
                            let _ = event_tx2
                                .send(StreamEvent::Text { text: text.clone() })
                                .await;
                        }
                        Chunk::Thinking(thinking) => {
                            let _ = event_tx2
                                .send(StreamEvent::Thinking {
                                    thinking: thinking.clone(),
                                })
                                .await;
                        }
                        Chunk::ToolUse { id, name, input } => {
                            let _ = event_tx2
                                .send(StreamEvent::ToolUse {
                                    id: id.clone(),
                                    name: name.clone(),
                                    input: input.clone(),
                                })
                                .await;
                            tool_uses.push((id, name, input));
                        }
                        Chunk::Done {
                            stop_reason,
                            content,
                        } => {
                            stop = stop_reason;
                            final_content = content;
                        }
                        Chunk::Error(msg) => {
                            let _ = event_tx2
                                .send(StreamEvent::Error {
                                    message: msg.clone(),
                                })
                                .await;
                        }
                    }
                }

                let _ = done_tx.send((tool_uses, final_content, stop));
            });

            // Drive the provider to completion.
            if let Err(e) = provider_handle.await {
                let _ = event_tx
                    .send(StreamEvent::Error {
                        message: format!("Provider error: {e}"),
                    })
                    .await;
                let _ = event_tx
                    .send(StreamEvent::Done {
                        reason: "error".to_string(),
                    })
                    .await;
                return Err(e);
            }

            // Wait for chunk processor to finish.
            let _ = chunk_processor.await;
            let (tool_uses, final_content, stop_reason) = done_rx.await.unwrap_or_else(|_| {
                (Vec::new(), Vec::new(), StopReason::EndTurn)
            });

            // Add assistant message to conversation.
            messages.push(Message {
                role: Role::Assistant,
                content: final_content,
            });

            // If no tool use, we're done.
            if stop_reason != StopReason::ToolUse || tool_uses.is_empty() {
                let reason = match stop_reason {
                    StopReason::EndTurn => "end_turn",
                    StopReason::MaxTokens => "max_tokens",
                    StopReason::ToolUse => "end_turn",
                };
                let _ = event_tx
                    .send(StreamEvent::Done {
                        reason: reason.to_string(),
                    })
                    .await;
                return Ok(());
            }

            // Execute tools and build tool result message.
            let mut tool_results = Vec::new();
            for (id, name, input) in tool_uses {
                info!(tool = %name, "Executing tool");
                let result = match self.tools.get(&name) {
                    Some(tool) => tool.call(input.clone()).await,
                    None => {
                        warn!(tool = %name, "Unknown tool");
                        crate::message::ToolResult::error(format!("Unknown tool: {name}"))
                    }
                };

                let content_text = result
                    .content
                    .iter()
                    .map(|c| match c {
                        ToolResultContent::Text { text } => text.as_str(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let _ = event_tx
                    .send(StreamEvent::ToolResult {
                        tool_use_id: id.clone(),
                        content: content_text,
                        is_error: result.is_error,
                    })
                    .await;

                tool_results.push(ContentBlock::ToolResult {
                    tool_use_id: id,
                    content: result.content,
                    is_error: if result.is_error { Some(true) } else { None },
                });
            }

            // Add tool results as user message.
            messages.push(Message {
                role: Role::User,
                content: tool_results,
            });
        }

        warn!("Agent reached max iterations");
        let _ = event_tx
            .send(StreamEvent::Done {
                reason: "max_iterations".to_string(),
            })
            .await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Chunk, Provider, Request, StopReason};
    use async_trait::async_trait;
    use serde_json::json;

    /// A mock provider that returns a fixed text response.
    struct TextProvider {
        text: String,
    }

    #[async_trait]
    impl Provider for TextProvider {
        async fn request(&self, _req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
            tx.send(Chunk::Text(self.text.clone())).await.unwrap();
            tx.send(Chunk::Done {
                stop_reason: StopReason::EndTurn,
                content: vec![ContentBlock::Text {
                    text: self.text.clone(),
                }],
            })
            .await
            .unwrap();
            Ok(())
        }
    }

    /// A mock provider that requests a tool call, then responds with text.
    struct ToolCallProvider;

    #[async_trait]
    impl Provider for ToolCallProvider {
        async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
            // If conversation already has a tool result, just respond with text.
            let has_tool_result = req.messages.iter().any(|m| {
                m.content
                    .iter()
                    .any(|c| matches!(c, ContentBlock::ToolResult { .. }))
            });

            if has_tool_result {
                tx.send(Chunk::Text("done".into())).await.unwrap();
                tx.send(Chunk::Done {
                    stop_reason: StopReason::EndTurn,
                    content: vec![ContentBlock::Text {
                        text: "done".into(),
                    }],
                })
                .await
                .unwrap();
            } else {
                let input = json!({"command": "echo test"});
                tx.send(Chunk::ToolUse {
                    id: "toolu_01".into(),
                    name: "bash".into(),
                    input: input.clone(),
                })
                .await
                .unwrap();
                tx.send(Chunk::Done {
                    stop_reason: StopReason::ToolUse,
                    content: vec![ContentBlock::ToolUse {
                        id: "toolu_01".into(),
                        name: "bash".into(),
                        input,
                    }],
                })
                .await
                .unwrap();
            }
            Ok(())
        }
    }

    /// A mock provider that always returns an error.
    struct ErrorProvider;

    #[async_trait]
    impl Provider for ErrorProvider {
        async fn request(&self, _req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
            let _ = tx.send(Chunk::Error("something broke".into())).await;
            anyhow::bail!("provider failed");
        }
    }

    /// A mock provider that always requests a tool call (never stops).
    struct InfiniteToolProvider;

    #[async_trait]
    impl Provider for InfiniteToolProvider {
        async fn request(&self, _req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
            let input = json!({"command": "echo loop"});
            let _ = tx
                .send(Chunk::ToolUse {
                    id: "toolu_01".into(),
                    name: "bash".into(),
                    input: input.clone(),
                })
                .await;
            let _ = tx
                .send(Chunk::Done {
                    stop_reason: StopReason::ToolUse,
                    content: vec![ContentBlock::ToolUse {
                        id: "toolu_01".into(),
                        name: "bash".into(),
                        input,
                    }],
                })
                .await;
            Ok(())
        }
    }

    async fn collect_events(mut rx: mpsc::Receiver<StreamEvent>) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        while let Some(e) = rx.recv().await {
            events.push(e);
        }
        events
    }

    #[tokio::test]
    async fn simple_text_response() {
        let provider = TextProvider {
            text: "Hello!".into(),
        };
        let tools = ToolRegistry::new();
        let agent = Agent::new(Box::new(provider), tools, "test".into(), "sys".into());

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("hi".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::Text { text } if text == "Hello!")));
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::Done { reason } if reason == "end_turn")));
    }

    #[tokio::test]
    async fn tool_call_and_response() {
        let provider = ToolCallProvider;
        let tools = ToolRegistry::with_defaults();
        let agent = Agent::new(Box::new(provider), tools, "test".into(), "sys".into());

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("run echo".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        // Should see: ToolUse → ToolResult → Text → Done
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolUse { name, .. } if name == "bash")));
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolResult { is_error, .. } if !is_error)));
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::Done { reason } if reason == "end_turn")));
    }

    #[tokio::test]
    async fn provider_error_emits_error_and_done() {
        let provider = ErrorProvider;
        let tools = ToolRegistry::new();
        let agent = Agent::new(Box::new(provider), tools, "test".into(), "sys".into());

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        let result = agent.run("hi".into(), tx).await;
        assert!(result.is_err());

        let events = handle.await.unwrap();
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::Error { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::Done { reason } if reason == "error")));
    }

    #[tokio::test]
    async fn max_iterations_terminates() {
        let provider = InfiniteToolProvider;
        let tools = ToolRegistry::with_defaults();
        let mut agent = Agent::new(Box::new(provider), tools, "test".into(), "sys".into());
        agent.max_iterations = 3;

        let (tx, rx) = mpsc::channel(256);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("loop".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::Done { reason } if reason == "max_iterations")));

        // Should have exactly 3 tool use events (one per iteration).
        let tool_use_count = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolUse { .. }))
            .count();
        assert_eq!(tool_use_count, 3);
    }
}
