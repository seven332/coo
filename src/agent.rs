use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use serde_json::json;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::message::{ContentBlock, Message, Role, StreamEvent, new_uuid};
use crate::provider::{Chunk, Provider, Request, ResponseMeta, ServerTool, StopReason};
use crate::tools::{BackgroundAgentResult, ToolContext, ToolRegistry, WorktreeInfo, git_cmd};
use std::collections::{HashMap, HashSet};

pub const DEFAULT_MAX_TOKENS: u32 = 32000;
pub const DEFAULT_MAX_ITERATIONS: usize = 100;
pub const DEFAULT_MAX_DEPTH: usize = 5;

/// Shared map of agent_id → queued messages, used by SendMessage to deliver to running agents.
pub type PendingMessagesMap = Arc<tokio::sync::Mutex<HashMap<String, Vec<String>>>>;

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
    pub cwd: Option<String>,
    /// Messages to prepend before the prompt (for fork optimization / prompt cache reuse).
    /// The cache_breakpoint is automatically set to the last prefix message.
    pub prefix_messages: Vec<Message>,
    /// Shared pending messages map from the parent, for receiving queued messages via SendMessage.
    /// If None, a fresh map is created in run_loop (typical for top-level agents).
    pub pending_messages: Option<PendingMessagesMap>,
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
            cwd: None,
            prefix_messages: Vec::new(),
            pending_messages: None,
        }
    }

    fn event_meta(&self) -> (String, String) {
        (self.session_id.clone(), new_uuid())
    }

    /// Inject a single background agent result as a user message.
    async fn inject_background_result(
        bg: BackgroundAgentResult,
        messages: &mut Vec<Message>,
        event_tx: &mpsc::Sender<StreamEvent>,
        session_id: &str,
    ) {
        let status = if bg.is_error { "failed" } else { "completed" };
        info!(
            agent_id = bg.agent_id,
            description = bg.description,
            status,
            "Background agent finished"
        );
        let notification = format!(
            "Background agent '{}' (id: {}) {status}:\n{}\n<usage>\ntotal_tokens: {}\ntool_uses: {}\nduration_ms: {}\n</usage>",
            bg.description,
            bg.agent_id,
            bg.result,
            bg.total_tokens,
            bg.tool_use_count,
            bg.duration_ms
        );
        let msg = Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: format!(
                    "<background-agent-notification>\n{notification}\n</background-agent-notification>"
                ),
            }],
        };
        let _ = event_tx
            .send(StreamEvent::User {
                message: msg.clone(),
                parent_tool_use_id: json!(null),
                session_id: session_id.to_string(),
                uuid: new_uuid(),
            })
            .await;
        messages.push(msg);
    }

    /// Drain completed background agent results (non-blocking).
    async fn drain_background_agents(
        background_rx: &mut mpsc::Receiver<BackgroundAgentResult>,
        messages: &mut Vec<Message>,
        event_tx: &mpsc::Sender<StreamEvent>,
        session_id: &str,
    ) {
        while let Ok(bg) = background_rx.try_recv() {
            Self::inject_background_result(bg, messages, event_tx, session_id).await;
        }
    }

    /// Block until all pending background agents complete, injecting each result.
    async fn wait_for_background_agents(
        background_rx: &mut mpsc::Receiver<BackgroundAgentResult>,
        pending: &AtomicUsize,
        messages: &mut Vec<Message>,
        event_tx: &mpsc::Sender<StreamEvent>,
        session_id: &str,
    ) {
        // Use timeout + recheck instead of snapshotting `pending`, because
        // drain (try_recv) may have consumed results whose bg tasks haven't
        // decremented `pending` yet — a snapshot would over-count.
        loop {
            if pending.load(Ordering::SeqCst) == 0 {
                break;
            }
            match tokio::time::timeout(std::time::Duration::from_millis(50), background_rx.recv())
                .await
            {
                Ok(Some(bg)) => {
                    Self::inject_background_result(bg, messages, event_tx, session_id).await;
                }
                Ok(None) => break,  // Channel closed.
                Err(_) => continue, // Timeout — recheck pending.
            }
        }
    }

    /// Abort all background agent tasks.
    async fn abort_background_agents(
        handles: &tokio::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>,
    ) {
        let handles = handles.lock().await;
        for handle in handles.iter() {
            handle.abort();
        }
    }

    /// Clean up worktrees registered by aborted background agents.
    async fn cleanup_background_worktrees(worktrees: &tokio::sync::Mutex<Vec<WorktreeInfo>>) {
        let worktrees = worktrees.lock().await;
        for wt in worktrees.iter() {
            let _ = git_cmd()
                .args(["worktree", "remove", "--force", &wt.path])
                .current_dir(&wt.git_root)
                .output()
                .await;
            let _ = git_cmd()
                .args(["branch", "-D", &wt.branch])
                .current_dir(&wt.git_root)
                .output()
                .await;
        }
    }

    /// Run the agent loop with a fresh prompt. Returns the conversation messages.
    /// If `prefix_messages` is set, they are prepended for prompt cache reuse.
    pub async fn run(
        &self,
        prompt: String,
        event_tx: mpsc::Sender<StreamEvent>,
    ) -> anyhow::Result<Vec<Message>> {
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

        let mut messages = self.prefix_messages.clone();
        messages.push(user_msg);
        self.run_loop(messages, event_tx).await
    }

    /// Resume the agent with existing conversation history plus a new message.
    pub async fn resume(
        &self,
        mut messages: Vec<Message>,
        new_message: String,
        event_tx: mpsc::Sender<StreamEvent>,
    ) -> anyhow::Result<Vec<Message>> {
        let user_msg = Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: new_message }],
        };

        let (sid, uid) = self.event_meta();
        let _ = event_tx
            .send(StreamEvent::User {
                message: user_msg.clone(),
                parent_tool_use_id: json!(null),
                session_id: sid,
                uuid: uid,
            })
            .await;

        messages.push(user_msg);
        self.run_loop(messages, event_tx).await
    }

    /// Core agent loop shared by run() and resume().
    async fn run_loop(
        &self,
        initial_messages: Vec<Message>,
        event_tx: mpsc::Sender<StreamEvent>,
    ) -> anyhow::Result<Vec<Message>> {
        let start = Instant::now();

        let mut messages = initial_messages;
        let tool_defs = self.tools.definitions();

        let (background_tx, mut background_rx) = mpsc::channel::<BackgroundAgentResult>(16);
        let pending_background = Arc::new(AtomicUsize::new(0));
        let background_handles = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let background_worktrees = Arc::new(tokio::sync::Mutex::new(Vec::<WorktreeInfo>::new()));
        let completed_agents = Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::<
            String,
            crate::tools::CompletedAgent,
        >::new()));
        let running_agents = Arc::new(tokio::sync::Mutex::new(HashSet::<String>::new()));
        let agent_name_registry =
            Arc::new(tokio::sync::Mutex::new(HashMap::<String, String>::new()));
        let pending_messages = self.pending_messages.clone().unwrap_or_else(|| {
            Arc::new(tokio::sync::Mutex::new(
                HashMap::<String, Vec<String>>::new(),
            ))
        });
        let parent_messages = Arc::new(tokio::sync::Mutex::new(Vec::new()));

        let tool_context = ToolContext {
            provider: self.provider.clone(),
            tools: self.tools.clone(),
            model: self.model.clone(),
            system: self.system.clone(),
            max_tokens: self.max_tokens,
            max_iterations: self.max_iterations,
            depth: self.depth,
            max_depth: self.max_depth,
            background_tx,
            pending_background: pending_background.clone(),
            background_handles: background_handles.clone(),
            background_worktrees: background_worktrees.clone(),
            completed_agents,
            running_agents,
            agent_name_registry,
            pending_messages,
            parent_messages: parent_messages.clone(),
            cwd: self.cwd.clone(),
        };

        let mut num_turns = 0;
        let mut iteration = 0;

        loop {
            if iteration >= self.max_iterations {
                break;
            }

            // Check for completed background agents and inject results into conversation.
            Self::drain_background_agents(
                &mut background_rx,
                &mut messages,
                &event_tx,
                &self.session_id,
            )
            .await;

            info!(iteration, "Agent loop iteration");
            iteration += 1;
            num_turns += 1;

            // Call LLM.
            let (chunk_tx, mut chunk_rx) = mpsc::channel::<Chunk>(64);

            // Set cache breakpoint at the last prefix message (if any) for prompt cache reuse.
            let cache_breakpoint = if !self.prefix_messages.is_empty() {
                Some(self.prefix_messages.len() - 1)
            } else {
                None
            };

            let req = Request {
                model: self.model.clone(),
                system: self.system.clone(),
                messages: messages.clone(),
                tools: tool_defs.clone(),
                server_tools: self.server_tools.clone(),
                max_tokens: self.max_tokens,
                cache_breakpoint,
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
                let orphaned = pending_background.load(Ordering::SeqCst);
                if orphaned > 0 {
                    warn!(orphaned, "Provider error — aborting background agents");
                    Self::abort_background_agents(&background_handles).await;
                    Self::cleanup_background_worktrees(&background_worktrees).await;
                }
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

            // MaxTokens: insert a user continuation message so the LLM picks up
            // where it left off. The Anthropic API requires alternating user/assistant
            // messages, so we cannot just `continue` with the last message being assistant.
            if stop_reason == StopReason::MaxTokens {
                info!("Response truncated (max_tokens) — continuing");
                let cont_msg = Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "Continue from where you left off.".to_string(),
                    }],
                };
                let (sid, uid) = self.event_meta();
                let _ = event_tx
                    .send(StreamEvent::User {
                        message: cont_msg.clone(),
                        parent_tool_use_id: json!(null),
                        session_id: sid,
                        uuid: uid,
                    })
                    .await;
                messages.push(cont_msg);
                continue;
            }

            // If no tool use, check if we can exit or need to wait for background agents.
            if stop_reason != StopReason::ToolUse || tool_uses.is_empty() {
                let msg_count_before = messages.len();

                // Wait for pending background agents first (if any).
                if pending_background.load(Ordering::SeqCst) > 0 {
                    info!("Waiting for background agents to complete");
                    Self::wait_for_background_agents(
                        &mut background_rx,
                        &pending_background,
                        &mut messages,
                        &event_tx,
                        &self.session_id,
                    )
                    .await;
                }

                // Drain any results sitting in the channel (already-completed agents).
                Self::drain_background_agents(
                    &mut background_rx,
                    &mut messages,
                    &event_tx,
                    &self.session_id,
                )
                .await;

                if messages.len() > msg_count_before {
                    // Background results were injected — let the LLM process them.
                    continue;
                }

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
                return Ok(messages);
            }

            // Update parent_messages snapshot before tool execution,
            // so sub-agents can inherit the current conversation for fork optimization.
            *parent_messages.lock().await = messages.clone();

            // Execute tools in parallel and build tool result message.
            let tool_futures: Vec<_> = tool_uses
                .into_iter()
                .map(|(id, name, input)| {
                    let tools = self.tools.clone();
                    let ctx = tool_context.clone();
                    async move {
                        info!(tool = %name, "Executing tool");
                        let result = match tools.get(&name) {
                            Some(tool) => tool.call(input.clone(), &ctx).await,
                            None => {
                                warn!(tool = %name, "Unknown tool");
                                crate::message::ToolResult::error(format!("Unknown tool: {name}"))
                            }
                        };
                        ContentBlock::ToolResult {
                            tool_use_id: id,
                            content: result.content,
                            is_error: if result.is_error { Some(true) } else { None },
                        }
                    }
                })
                .collect();
            let tool_results = futures::future::join_all(tool_futures).await;

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

            // Drain pending messages queued by SendMessage for this agent.
            let drained = {
                let mut pm = tool_context.pending_messages.lock().await;
                pm.remove(&self.session_id).unwrap_or_default()
            };
            if !drained.is_empty() {
                info!(
                    count = drained.len(),
                    "Injecting pending messages from SendMessage"
                );
                let text = drained
                    .into_iter()
                    .map(|m| format!("<queued-message>\n{m}\n</queued-message>"))
                    .collect::<Vec<_>>()
                    .join("\n");
                let pending_msg = Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text { text }],
                };
                let (sid, uid) = self.event_meta();
                let _ = event_tx
                    .send(StreamEvent::User {
                        message: pending_msg.clone(),
                        parent_tool_use_id: json!(null),
                        session_id: sid,
                        uuid: uid,
                    })
                    .await;
                messages.push(pending_msg);
            }
        }

        let orphaned = pending_background.load(Ordering::SeqCst);
        if orphaned > 0 {
            warn!(orphaned, "Max iterations — aborting background agents");
            Self::abort_background_agents(&background_handles).await;
            Self::cleanup_background_worktrees(&background_worktrees).await;
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
        Ok(messages)
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
    async fn prefix_messages_sent_to_provider() {
        // Provider that reports how many messages it received.
        struct MessageCountProvider;

        #[async_trait]
        impl Provider for MessageCountProvider {
            async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let count = req.messages.len();
                let bp = req
                    .cache_breakpoint
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "none".into());
                let text = format!("messages:{count} cache_breakpoint:{bp}");
                let _ = tx.send(Chunk::Text(text.clone())).await;
                let _ = tx
                    .send(Chunk::Done {
                        stop_reason: StopReason::EndTurn,
                        content: vec![ContentBlock::Text { text }],
                    })
                    .await;
                Ok(())
            }
        }

        let mut agent = make_agent(MessageCountProvider, ToolRegistry::new());
        // Add 2 prefix messages (simulating parent conversation).
        agent.prefix_messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "parent msg 1".into(),
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "parent reply".into(),
                }],
            },
        ];

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("sub-agent prompt".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        // 2 prefix + 1 new prompt = 3 messages.
        assert!(has_assistant_with_text(&events, "messages:3"));
        // cache_breakpoint should be at index 1 (last prefix message).
        assert!(has_assistant_with_text(&events, "cache_breakpoint:1"));
    }

    #[tokio::test]
    async fn no_prefix_no_cache_breakpoint() {
        struct BreakpointCheckProvider;

        #[async_trait]
        impl Provider for BreakpointCheckProvider {
            async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let bp = req
                    .cache_breakpoint
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "none".into());
                let text = format!("cache_breakpoint:{bp}");
                let _ = tx.send(Chunk::Text(text.clone())).await;
                let _ = tx
                    .send(Chunk::Done {
                        stop_reason: StopReason::EndTurn,
                        content: vec![ContentBlock::Text { text }],
                    })
                    .await;
                Ok(())
            }
        }

        let agent = make_agent(BreakpointCheckProvider, ToolRegistry::new());

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("test".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        assert!(has_assistant_with_text(&events, "cache_breakpoint:none"));
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

    #[tokio::test]
    async fn background_agent_waited_before_exit() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Provider that:
        // 1st call: requests agent tool with run_in_background
        // 2nd call (after tool result): end_turn (but bg agent still running)
        // 3rd call (after bg notification): end_turn with "saw background"
        struct BgTestProvider {
            call_count: AtomicUsize,
        }

        #[async_trait]
        impl Provider for BgTestProvider {
            async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);

                // Check if messages contain a background notification.
                let has_bg_notification = req.messages.iter().any(|m| {
                    m.content.iter().any(|c| match c {
                        ContentBlock::Text { text } => {
                            text.contains("background-agent-notification")
                        }
                        _ => false,
                    })
                });

                if has_bg_notification {
                    // 3rd call: we got the background notification.
                    let _ = tx.send(Chunk::Text("saw background result".into())).await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::EndTurn,
                            content: vec![ContentBlock::Text {
                                text: "saw background result".into(),
                            }],
                        })
                        .await;
                } else if n == 0 {
                    // 1st call: request agent tool with run_in_background.
                    let input = json!({
                        "prompt": "background work",
                        "run_in_background": true,
                        "description": "bg test"
                    });
                    let _ = tx
                        .send(Chunk::ToolUse {
                            id: "t1".into(),
                            name: "agent".into(),
                            input: input.clone(),
                        })
                        .await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::ToolUse,
                            content: vec![ContentBlock::ToolUse {
                                id: "t1".into(),
                                name: "agent".into(),
                                input,
                            }],
                        })
                        .await;
                } else {
                    // 2nd call: end_turn (bg agent may still be running).
                    let _ = tx.send(Chunk::Text("waiting...".into())).await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::EndTurn,
                            content: vec![ContentBlock::Text {
                                text: "waiting...".into(),
                            }],
                        })
                        .await;
                }
                Ok(())
            }
        }

        let provider = BgTestProvider {
            call_count: AtomicUsize::new(0),
        };
        let agent = make_agent(provider, ToolRegistry::with_defaults());

        let (tx, rx) = mpsc::channel(256);
        let handle = tokio::spawn(async move { collect_events(rx).await });

        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            agent.run("go".into(), tx),
        )
        .await
        .expect("agent should complete within timeout")
        .unwrap();

        let events = handle.await.unwrap();

        // The agent should have seen the background notification and responded.
        assert!(
            has_assistant_with_text(&events, "saw background result"),
            "agent should process background agent result before exiting"
        );
        assert!(has_result(&events, "success"));

        // Should have a user message with the background notification.
        let has_bg_notification = events.iter().any(|e| {
            if let StreamEvent::User { message, .. } = e {
                message.content.iter().any(|c| match c {
                    ContentBlock::Text { text } => text.contains("background-agent-notification"),
                    _ => false,
                })
            } else {
                false
            }
        });
        assert!(
            has_bg_notification,
            "should have background agent notification in events"
        );
    }

    #[tokio::test]
    async fn multiple_background_agents_all_waited() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Provider that spawns 2 background agents, then waits for both.
        struct MultiBgProvider {
            call_count: AtomicUsize,
        }

        #[async_trait]
        impl Provider for MultiBgProvider {
            async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);

                let bg_notification_count = req
                    .messages
                    .iter()
                    .flat_map(|m| &m.content)
                    .filter(|c| match c {
                        ContentBlock::Text { text } => {
                            text.contains("background-agent-notification")
                        }
                        _ => false,
                    })
                    .count();

                if bg_notification_count >= 2 {
                    // Got both notifications.
                    let _ = tx.send(Chunk::Text("saw both".into())).await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::EndTurn,
                            content: vec![ContentBlock::Text {
                                text: "saw both".into(),
                            }],
                        })
                        .await;
                } else if n == 0 {
                    // First call: spawn 2 background agents.
                    for i in 1..=2 {
                        let input = json!({
                            "prompt": format!("bg task {i}"),
                            "run_in_background": true,
                            "description": format!("bg {i}")
                        });
                        let _ = tx
                            .send(Chunk::ToolUse {
                                id: format!("t{i}"),
                                name: "agent".into(),
                                input: input.clone(),
                            })
                            .await;
                    }
                    let input1 = json!({"prompt": "bg task 1", "run_in_background": true, "description": "bg 1"});
                    let input2 = json!({"prompt": "bg task 2", "run_in_background": true, "description": "bg 2"});
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::ToolUse,
                            content: vec![
                                ContentBlock::ToolUse {
                                    id: "t1".into(),
                                    name: "agent".into(),
                                    input: input1,
                                },
                                ContentBlock::ToolUse {
                                    id: "t2".into(),
                                    name: "agent".into(),
                                    input: input2,
                                },
                            ],
                        })
                        .await;
                } else {
                    // Waiting for bg agents.
                    let _ = tx.send(Chunk::Text("waiting".into())).await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::EndTurn,
                            content: vec![ContentBlock::Text {
                                text: "waiting".into(),
                            }],
                        })
                        .await;
                }
                Ok(())
            }
        }

        let provider = MultiBgProvider {
            call_count: AtomicUsize::new(0),
        };
        let agent = make_agent(provider, ToolRegistry::with_defaults());

        let (tx, rx) = mpsc::channel(256);
        let handle = tokio::spawn(async move { collect_events(rx).await });

        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            agent.run("go".into(), tx),
        )
        .await
        .expect("should complete within timeout")
        .unwrap();

        let events = handle.await.unwrap();

        assert!(
            has_assistant_with_text(&events, "saw both"),
            "agent should process both background results"
        );

        let bg_notification_count = events
            .iter()
            .filter(|e| {
                if let StreamEvent::User { message, .. } = e {
                    message.content.iter().any(|c| match c {
                        ContentBlock::Text { text } => {
                            text.contains("background-agent-notification")
                        }
                        _ => false,
                    })
                } else {
                    false
                }
            })
            .count();
        assert_eq!(
            bg_notification_count, 2,
            "should have 2 background notifications"
        );
    }

    #[tokio::test]
    async fn background_agents_aborted_on_max_iterations() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Provider that spawns a bg agent, then always requests tool use to burn iterations.
        struct BurnIterProvider {
            call_count: AtomicUsize,
        }

        #[async_trait]
        impl Provider for BurnIterProvider {
            async fn request(&self, _req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    // Spawn bg agent.
                    let input = json!({
                        "prompt": "long task",
                        "run_in_background": true,
                    });
                    let _ = tx
                        .send(Chunk::ToolUse {
                            id: "t1".into(),
                            name: "agent".into(),
                            input: input.clone(),
                        })
                        .await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::ToolUse,
                            content: vec![ContentBlock::ToolUse {
                                id: "t1".into(),
                                name: "agent".into(),
                                input,
                            }],
                        })
                        .await;
                } else {
                    // Keep requesting tool use to burn iterations.
                    let input = json!({"command": "echo burn"});
                    let _ = tx
                        .send(Chunk::ToolUse {
                            id: format!("b{n}"),
                            name: "bash".into(),
                            input: input.clone(),
                        })
                        .await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::ToolUse,
                            content: vec![ContentBlock::ToolUse {
                                id: format!("b{n}"),
                                name: "bash".into(),
                                input,
                            }],
                        })
                        .await;
                }
                Ok(())
            }
        }

        let provider = BurnIterProvider {
            call_count: AtomicUsize::new(0),
        };
        let mut agent = make_agent(provider, ToolRegistry::with_defaults());
        agent.max_iterations = 3;

        let (tx, rx) = mpsc::channel(512);
        let handle = tokio::spawn(async move { collect_events(rx).await });

        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            agent.run("go".into(), tx),
        )
        .await
        .expect("should complete within timeout")
        .unwrap();

        let events = handle.await.unwrap();

        assert!(
            has_result(&events, "error_max_turns"),
            "should hit max iterations"
        );
    }

    #[tokio::test]
    async fn max_tokens_continues_conversation() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Provider that returns MaxTokens on first call, then EndTurn on second.
        struct MaxTokensProvider {
            call_count: AtomicUsize,
        }

        #[async_trait]
        impl Provider for MaxTokensProvider {
            async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                // Verify messages alternate user/assistant (as Anthropic API requires).
                for pair in req.messages.windows(2) {
                    assert_ne!(pair[0].role, pair[1].role, "messages must alternate roles");
                }
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    // First call: truncated response.
                    let _ = tx.send(Chunk::Text("partial".into())).await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::MaxTokens,
                            content: vec![ContentBlock::Text {
                                text: "partial".into(),
                            }],
                        })
                        .await;
                } else {
                    // Second call: complete response.
                    let _ = tx.send(Chunk::Text(" complete".into())).await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::EndTurn,
                            content: vec![ContentBlock::Text {
                                text: " complete".into(),
                            }],
                        })
                        .await;
                }
                Ok(())
            }
        }

        let provider = MaxTokensProvider {
            call_count: AtomicUsize::new(0),
        };
        let agent = make_agent(provider, ToolRegistry::new());

        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move { collect_events(rx).await });
        agent.run("test".into(), tx).await.unwrap();
        let events = handle.await.unwrap();

        // Should have 2 assistant messages (one truncated, one complete).
        let assistant_count = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Assistant { .. }))
            .count();
        assert_eq!(assistant_count, 2, "should continue after max_tokens");

        // Should succeed, not error.
        assert!(has_result(&events, "success"));

        // Provider should have been called twice.
        assert!(
            has_assistant_with_text(&events, "complete"),
            "should have the continuation response"
        );

        // Should have a user continuation message injected between the two assistant messages.
        let has_continuation = events.iter().any(|e| {
            if let StreamEvent::User { message, .. } = e {
                message.content.iter().any(|c| match c {
                    ContentBlock::Text { text } => text.contains("Continue"),
                    _ => false,
                })
            } else {
                false
            }
        });
        assert!(
            has_continuation,
            "should inject a user continuation message after max_tokens"
        );
    }

    #[tokio::test]
    async fn parallel_tool_execution() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Instant;

        // Provider that requests 3 bash tool calls, then ends.
        struct ParallelToolProvider {
            call_count: AtomicUsize,
        }

        #[async_trait]
        impl Provider for ParallelToolProvider {
            async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);
                let has_tool_result = req.messages.iter().any(|m| {
                    m.content
                        .iter()
                        .any(|c| matches!(c, ContentBlock::ToolResult { .. }))
                });

                if has_tool_result || n > 0 {
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
                    // Request 3 sleep commands — should run in parallel.
                    for i in 1..=3 {
                        let input = json!({"command": "sleep 0.3 && echo ok"});
                        let _ = tx
                            .send(Chunk::ToolUse {
                                id: format!("t{i}"),
                                name: "bash".into(),
                                input: input.clone(),
                            })
                            .await;
                    }
                    let input = json!({"command": "sleep 0.3 && echo ok"});
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::ToolUse,
                            content: vec![
                                ContentBlock::ToolUse {
                                    id: "t1".into(),
                                    name: "bash".into(),
                                    input: input.clone(),
                                },
                                ContentBlock::ToolUse {
                                    id: "t2".into(),
                                    name: "bash".into(),
                                    input: input.clone(),
                                },
                                ContentBlock::ToolUse {
                                    id: "t3".into(),
                                    name: "bash".into(),
                                    input,
                                },
                            ],
                        })
                        .await;
                }
                Ok(())
            }
        }

        let provider = ParallelToolProvider {
            call_count: AtomicUsize::new(0),
        };
        let agent = make_agent(provider, ToolRegistry::with_defaults());

        let (tx, rx) = mpsc::channel(256);
        let handle = tokio::spawn(async move { collect_events(rx).await });

        let start = Instant::now();
        agent.run("parallel test".into(), tx).await.unwrap();
        let elapsed = start.elapsed();
        let events = handle.await.unwrap();

        // All 3 tool results should be present.
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
        assert_eq!(tool_result_count, 3, "should have 3 tool results");

        // If run in parallel, 3x 300ms should take < 900ms.
        // Allow generous margin but must be significantly less than sequential.
        assert!(
            elapsed.as_millis() < 800,
            "parallel execution should take < 800ms, took {}ms",
            elapsed.as_millis()
        );

        assert!(has_result(&events, "success"));
    }
}
