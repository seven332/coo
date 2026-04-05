use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::message::{ContentBlock, Message, Role, StreamEvent, new_uuid};
use crate::provider::{Chunk, Provider, Request, ResponseMeta, ServerTool, StopReason};
use crate::tools::{ToolContext, ToolRegistry};

pub const DEFAULT_MAX_TOKENS: u32 = 32000;
pub const DEFAULT_MAX_ITERATIONS: usize = 100;
pub const DEFAULT_MAX_DEPTH: usize = 5;

pub struct Agent {
    provider: Arc<dyn Provider>,
    tools: Arc<ToolRegistry>,
    pub model: String,
    system: String,
    pub max_tokens: u32,
    pub max_iterations: usize,
    pub server_tools: Vec<ServerTool>,
    pub depth: usize,
    pub max_depth: usize,
    pub session_id: String,
}

impl Agent {
    pub fn new(
        provider: Arc<dyn Provider>,
        tools: Arc<ToolRegistry>,
        model: String,
        system: String,
    ) -> Self {
        Self {
            provider,
            tools,
            model,
            system,
            max_tokens: DEFAULT_MAX_TOKENS,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            server_tools: Vec::new(),
            depth: 0,
            max_depth: DEFAULT_MAX_DEPTH,
            session_id: new_uuid(),
        }
    }

    fn event_meta(&self) -> (String, String) {
        (self.session_id.clone(), new_uuid())
    }

    /// Run the agent loop. Streams events to `event_tx`.
    /// Takes an initial user prompt and runs until completion.
    pub async fn run(
        &self,
        prompt: String,
        event_tx: mpsc::Sender<StreamEvent>,
    ) -> anyhow::Result<()> {
        let start = Instant::now();

        let user_msg = Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: prompt }],
        };

        // Emit user message.
        let (sid, uid) = self.event_meta();
        let _ = event_tx
            .send(StreamEvent::User {
                message: user_msg.clone(),
                parent_tool_use_id: json!(null),
                session_id: sid,
                uuid: uid,
            })
            .await;

        let mut messages = vec![user_msg];
        let tool_defs = self.tools.definitions();

        let tool_context = ToolContext {
            provider: self.provider.clone(),
            tools: self.tools.clone(),
            model: self.model.clone(),
            system: self.system.clone(),
            max_tokens: self.max_tokens,
            max_iterations: self.max_iterations,
            depth: self.depth,
            max_depth: self.max_depth,
        };

        let mut num_turns = 0;

        for iteration in 0..self.max_iterations {
            info!(iteration, "Agent loop iteration");
            num_turns += 1;

            // Call LLM.
            let (chunk_tx, mut chunk_rx) = mpsc::channel::<Chunk>(64);

            let req = Request {
                model: self.model.clone(),
                system: self.system.clone(),
                messages: messages.clone(),
                tools: tool_defs.clone(),
                server_tools: self.server_tools.clone(),
                max_tokens: self.max_tokens,
            };

            let provider_handle = {
                let provider = &self.provider;
                let chunk_tx = chunk_tx;
                provider.request(req, chunk_tx)
            };

            // Process chunks concurrently with the provider streaming.
            let event_tx2 = event_tx.clone();
            let session_id = self.session_id.clone();
            let (done_tx, done_rx) = tokio::sync::oneshot::channel();

            let chunk_processor = tokio::spawn(async move {
                let mut tool_uses = Vec::new();
                let mut final_content = Vec::new();
                let mut stop = StopReason::EndTurn;
                let mut response_meta: Option<ResponseMeta> = None;

                while let Some(chunk) = chunk_rx.recv().await {
                    match chunk {
                        Chunk::Meta(m) => {
                            response_meta = Some(m);
                        }
                        Chunk::Text(text) => {
                            let _ = event_tx2
                                .send(StreamEvent::RawEvent {
                                    event: json!({
                                        "type": "content_block_delta",
                                        "delta": {"type": "text_delta", "text": text}
                                    }),
                                    parent_tool_use_id: json!(null),
                                    session_id: session_id.clone(),
                                    uuid: new_uuid(),
                                })
                                .await;
                        }
                        Chunk::Thinking(thinking) => {
                            let _ = event_tx2
                                .send(StreamEvent::RawEvent {
                                    event: json!({
                                        "type": "content_block_delta",
                                        "delta": {"type": "thinking_delta", "thinking": thinking}
                                    }),
                                    parent_tool_use_id: json!(null),
                                    session_id: session_id.clone(),
                                    uuid: new_uuid(),
                                })
                                .await;
                        }
                        Chunk::ToolUse { id, name, input } => {
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
                                .send(StreamEvent::RawEvent {
                                    event: json!({"type": "error", "error": {"message": msg}}),
                                    parent_tool_use_id: json!(null),
                                    session_id: session_id.clone(),
                                    uuid: new_uuid(),
                                })
                                .await;
                        }
                    }
                }

                let _ = done_tx.send((tool_uses, final_content, stop, response_meta));
            });

            // Drive the provider to completion.
            if let Err(e) = provider_handle.await {
                let _ = chunk_processor.await;
                let duration_ms = start.elapsed().as_millis() as u64;
                let (sid, uid) = self.event_meta();
                let _ = event_tx
                    .send(StreamEvent::Result {
                        subtype: "error_during_execution".to_string(),
                        duration_ms,
                        is_error: true,
                        num_turns,
                        result: None,
                        stop_reason: None,
                        session_id: sid,
                        uuid: uid,
                    })
                    .await;
                return Err(e);
            }

            // Wait for chunk processor to finish.
            let _ = chunk_processor.await;
            let (tool_uses, final_content, stop_reason, response_meta) = done_rx
                .await
                .unwrap_or_else(|_| (Vec::new(), Vec::new(), StopReason::EndTurn, None));

            // Add assistant message to conversation and emit event.
            let assistant_msg = Message {
                role: Role::Assistant,
                content: final_content,
            };

            // Build full API-format message for the event.
            let stop_str = match stop_reason {
                StopReason::EndTurn => "end_turn",
                StopReason::ToolUse => "tool_use",
                StopReason::MaxTokens => "max_tokens",
            };
            let mut api_message = json!({
                "type": "message",
                "role": "assistant",
                "content": serde_json::to_value(&assistant_msg.content).unwrap_or_default(),
                "model": response_meta.as_ref().map(|m| m.model.as_str()).unwrap_or(&self.model),
                "stop_reason": stop_str,
            });
            if let Some(ref meta) = response_meta {
                if !meta.message_id.is_empty() {
                    api_message["id"] = json!(meta.message_id);
                }
                if !meta.usage.is_null() {
                    api_message["usage"] = meta.usage.clone();
                }
            }

            let (sid, uid) = self.event_meta();
            let _ = event_tx
                .send(StreamEvent::Assistant {
                    message: api_message,
                    parent_tool_use_id: json!(null),
                    session_id: sid,
                    uuid: uid,
                })
                .await;
            messages.push(assistant_msg);

            // If no tool use, we're done.
            if stop_reason != StopReason::ToolUse || tool_uses.is_empty() {
                let result_text = messages
                    .last()
                    .map(|m| {
                        m.content
                            .iter()
                            .filter_map(|c| match c {
                                ContentBlock::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("")
                    })
                    .unwrap_or_default();

                let duration_ms = start.elapsed().as_millis() as u64;
                let (sid, uid) = self.event_meta();
                let _ = event_tx
                    .send(StreamEvent::Result {
                        subtype: "success".to_string(),
                        duration_ms,
                        is_error: false,
                        num_turns,
                        result: Some(result_text),
                        stop_reason: Some(stop_str.to_string()),
                        session_id: sid,
                        uuid: uid,
                    })
                    .await;
                return Ok(());
            }

            // Execute tools and build tool result message.
            let mut tool_results = Vec::new();
            for (id, name, input) in tool_uses {
                info!(tool = %name, "Executing tool");
                let result = match self.tools.get(&name) {
                    Some(tool) => tool.call(input.clone(), &tool_context).await,
                    None => {
                        warn!(tool = %name, "Unknown tool");
                        crate::message::ToolResult::error(format!("Unknown tool: {name}"))
                    }
                };

                tool_results.push(ContentBlock::ToolResult {
                    tool_use_id: id,
                    content: result.content,
                    is_error: if result.is_error { Some(true) } else { None },
                });
            }

            // Add tool results as user message and emit event.
            let tool_result_msg = Message {
                role: Role::User,
                content: tool_results,
            };
            let (sid, uid) = self.event_meta();
            let _ = event_tx
                .send(StreamEvent::User {
                    message: tool_result_msg.clone(),
                    parent_tool_use_id: json!(null),
                    session_id: sid,
                    uuid: uid,
                })
                .await;
            messages.push(tool_result_msg);
        }

        warn!("Agent reached max iterations");
        let duration_ms = start.elapsed().as_millis() as u64;
        let (sid, uid) = self.event_meta();
        let _ = event_tx
            .send(StreamEvent::Result {
                subtype: "error_max_turns".to_string(),
                duration_ms,
                is_error: true,
                num_turns,
                result: None,
                stop_reason: None,
                session_id: sid,
                uuid: uid,
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

    fn make_agent<P: Provider + 'static>(provider: P, tools: ToolRegistry) -> Agent {
        Agent::new(
            Arc::new(provider),
            Arc::new(tools),
            "test".into(),
            "sys".into(),
        )
    }

    fn has_assistant_with_text(events: &[StreamEvent], expected: &str) -> bool {
        events.iter().any(|e| {
            if let StreamEvent::Assistant { message, .. } = e {
                message
                    .get("content")
                    .and_then(|c| c.as_array())
                    .is_some_and(|blocks| {
                        blocks.iter().any(|b| {
                            b.get("type").and_then(|t| t.as_str()) == Some("text")
                                && b.get("text")
                                    .and_then(|t| t.as_str())
                                    .is_some_and(|t| t.contains(expected))
                        })
                    })
            } else {
                false
            }
        })
    }

    fn has_result(events: &[StreamEvent], expected_subtype: &str) -> bool {
        events.iter().any(
            |e| matches!(e, StreamEvent::Result { subtype, .. } if subtype == expected_subtype),
        )
    }

    fn has_user_with_tool_result(events: &[StreamEvent]) -> bool {
        events.iter().any(|e| {
            if let StreamEvent::User { message, .. } = e {
                message
                    .content
                    .iter()
                    .any(|c| matches!(c, ContentBlock::ToolResult { .. }))
            } else {
                false
            }
        })
    }

    #[tokio::test]
    async fn simple_text_response() {
        let agent = make_agent(
            TextProvider {
                text: "Hello!".into(),
            },
            ToolRegistry::new(),
        );

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("hi".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        assert!(has_assistant_with_text(&events, "Hello!"));
        assert!(has_result(&events, "success"));
    }

    #[tokio::test]
    async fn tool_call_and_response() {
        let agent = make_agent(ToolCallProvider, ToolRegistry::with_defaults());

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("run echo".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        // Should have assistant messages and user tool result messages.
        assert!(has_user_with_tool_result(&events));
        assert!(has_result(&events, "success"));
    }

    #[tokio::test]
    async fn provider_error_emits_error_result() {
        let agent = make_agent(ErrorProvider, ToolRegistry::new());

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        let result = agent.run("hi".into(), tx).await;
        assert!(result.is_err());

        let events = handle.await.unwrap();
        assert!(has_result(&events, "error_during_execution"));
    }

    #[tokio::test]
    async fn max_iterations_terminates() {
        let mut agent = make_agent(InfiniteToolProvider, ToolRegistry::with_defaults());
        agent.max_iterations = 3;

        let (tx, rx) = mpsc::channel(256);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("loop".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        assert!(has_result(&events, "error_max_turns"));

        // Should have 3 assistant messages (one per iteration).
        let assistant_count = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Assistant { .. }))
            .count();
        assert_eq!(assistant_count, 3);
    }

    #[tokio::test]
    async fn event_ordering() {
        let agent = make_agent(ToolCallProvider, ToolRegistry::with_defaults());

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("test".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        // Collect event type names in order.
        let types: Vec<&str> = events
            .iter()
            .map(|e| match e {
                StreamEvent::System { .. } => "system",
                StreamEvent::User { .. } => "user",
                StreamEvent::Assistant { .. } => "assistant",
                StreamEvent::RawEvent { .. } => "stream_event",
                StreamEvent::Result { .. } => "result",
            })
            .collect();

        // First event must be User (prompt).
        assert_eq!(types[0], "user");
        // Last event must be Result.
        assert_eq!(types[types.len() - 1], "result");
        // Each Assistant must be followed by either User (tool results) or Result.
        for (i, t) in types.iter().enumerate() {
            if *t == "assistant" && i + 1 < types.len() {
                assert!(
                    types[i + 1] == "user" || types[i + 1] == "result",
                    "assistant at {i} followed by {} instead of user/result",
                    types[i + 1]
                );
            }
        }
    }

    #[tokio::test]
    async fn session_id_consistent() {
        let agent = make_agent(TextProvider { text: "hi".into() }, ToolRegistry::new());

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("test".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        let session_ids: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::User { session_id, .. }
                | StreamEvent::Assistant { session_id, .. }
                | StreamEvent::RawEvent { session_id, .. }
                | StreamEvent::Result { session_id, .. } => Some(session_id.as_str()),
                _ => None,
            })
            .collect();

        assert!(!session_ids.is_empty());
        let first = session_ids[0];
        assert!(
            session_ids.iter().all(|s| *s == first),
            "all events must share the same session_id"
        );
    }

    #[tokio::test]
    async fn thinking_forwarded_as_raw_event() {
        struct ThinkingProvider;

        #[async_trait]
        impl Provider for ThinkingProvider {
            async fn request(&self, _req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let _ = tx.send(Chunk::Thinking("hmm".into())).await;
                let _ = tx.send(Chunk::Text("ok".into())).await;
                let _ = tx
                    .send(Chunk::Done {
                        stop_reason: StopReason::EndTurn,
                        content: vec![
                            ContentBlock::Thinking {
                                thinking: "hmm".into(),
                            },
                            ContentBlock::Text { text: "ok".into() },
                        ],
                    })
                    .await;
                Ok(())
            }
        }

        let agent = make_agent(ThinkingProvider, ToolRegistry::new());
        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("think".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        let has_thinking_event = events.iter().any(|e| {
            if let StreamEvent::RawEvent { event, .. } = e {
                event
                    .get("delta")
                    .and_then(|d| d.get("type"))
                    .is_some_and(|t| t == "thinking_delta")
            } else {
                false
            }
        });
        assert!(
            has_thinking_event,
            "thinking should be forwarded as RawEvent"
        );
    }

    #[tokio::test]
    async fn empty_assistant_response() {
        struct EmptyProvider;

        #[async_trait]
        impl Provider for EmptyProvider {
            async fn request(&self, _req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let _ = tx
                    .send(Chunk::Done {
                        stop_reason: StopReason::EndTurn,
                        content: vec![],
                    })
                    .await;
                Ok(())
            }
        }

        let agent = make_agent(EmptyProvider, ToolRegistry::new());
        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("test".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        // Should still get Assistant and Result events.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::Assistant { .. }))
        );
        assert!(has_result(&events, "success"));

        // Result text should be empty.
        let result_text = events.iter().find_map(|e| {
            if let StreamEvent::Result {
                result: Some(text), ..
            } = e
            {
                Some(text.as_str())
            } else {
                None
            }
        });
        assert_eq!(result_text, Some(""));
    }

    #[tokio::test]
    async fn multiple_tool_calls_single_iteration() {
        struct MultiToolProvider;

        #[async_trait]
        impl Provider for MultiToolProvider {
            async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let has_tool_result = req.messages.iter().any(|m| {
                    m.content
                        .iter()
                        .any(|c| matches!(c, ContentBlock::ToolResult { .. }))
                });

                if has_tool_result {
                    let _ = tx.send(Chunk::Text("done".into())).await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::EndTurn,
                            content: vec![ContentBlock::Text {
                                text: "done".into(),
                            }],
                        })
                        .await;
                } else {
                    let input1 = json!({"command": "echo a"});
                    let input2 = json!({"command": "echo b"});
                    let _ = tx
                        .send(Chunk::ToolUse {
                            id: "t1".into(),
                            name: "bash".into(),
                            input: input1.clone(),
                        })
                        .await;
                    let _ = tx
                        .send(Chunk::ToolUse {
                            id: "t2".into(),
                            name: "bash".into(),
                            input: input2.clone(),
                        })
                        .await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::ToolUse,
                            content: vec![
                                ContentBlock::ToolUse {
                                    id: "t1".into(),
                                    name: "bash".into(),
                                    input: input1,
                                },
                                ContentBlock::ToolUse {
                                    id: "t2".into(),
                                    name: "bash".into(),
                                    input: input2,
                                },
                            ],
                        })
                        .await;
                }
                Ok(())
            }
        }

        let agent = make_agent(MultiToolProvider, ToolRegistry::with_defaults());
        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("run two".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        // Tool result user message should have 2 ToolResult blocks.
        let tool_result_count = events
            .iter()
            .filter_map(|e| {
                if let StreamEvent::User { message, .. } = e {
                    Some(
                        message
                            .content
                            .iter()
                            .filter(|c| matches!(c, ContentBlock::ToolResult { .. }))
                            .count(),
                    )
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);
        assert_eq!(
            tool_result_count, 2,
            "should have 2 tool results in one user message"
        );
        assert!(has_result(&events, "success"));
    }
}
