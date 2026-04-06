use std::sync::atomic::Ordering;
use std::time::Instant;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::info;

use crate::agent::Agent;
use crate::message::{StreamEvent, ToolResult};

use super::{BackgroundAgentResult, Tool, ToolContext};

pub struct AgentTool;

/// Aborts the associated task when dropped.
/// Ensures inner spawned tasks are cleaned up if the outer task is cancelled.
struct AbortOnDrop(tokio::task::AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    run_in_background: bool,
}

/// Collect text output, token usage, tool use count, and error status from agent events.
/// Returns (output, total_tokens, tool_use_count, had_error).
async fn collect_agent_output(
    event_rx: &mut mpsc::Receiver<StreamEvent>,
) -> (String, u64, usize, bool) {
    let mut output = String::new();
    let mut total_tokens = 0u64;
    let mut tool_use_count = 0usize;
    let mut had_error = false;

    while let Some(event) = event_rx.recv().await {
        match event {
            StreamEvent::Assistant { ref message, .. } => {
                if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
                    for block in content {
                        if block.get("type").and_then(|t| t.as_str()) == Some("text")
                            && let Some(text) = block.get("text").and_then(|t| t.as_str())
                        {
                            output.push_str(text);
                        }
                        if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                            tool_use_count += 1;
                        }
                    }
                }
                if let Some(usage) = message.get("usage") {
                    if let Some(input_t) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                        total_tokens += input_t;
                    }
                    if let Some(output_t) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                        total_tokens += output_t;
                    }
                }
            }
            StreamEvent::Result {
                is_error: true,
                subtype,
                ..
            } => {
                had_error = true;
                output = subtype;
            }
            _ => {}
        }
    }

    (output, total_tokens, tool_use_count, had_error)
}

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        "agent"
    }

    fn description(&self) -> &str {
        "Spawn a sub-agent to handle a complex task. The sub-agent has access to the same tools \
         and runs independently. Use this for tasks that benefit from a separate context."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the sub-agent to perform"
                },
                "description": {
                    "type": "string",
                    "description": "A short (3-5 word) description of the task"
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override (e.g. 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001')"
                },
                "system": {
                    "type": "string",
                    "description": "Optional system prompt override for the sub-agent"
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Set to true to run the agent in the background. Returns immediately with agent ID."
                }
            },
            "required": ["prompt"]
        })
    }

    async fn call(&self, input: serde_json::Value, context: &ToolContext) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        if context.depth >= context.max_depth {
            return ToolResult::error(format!("Max agent depth {} reached", context.max_depth));
        }

        let description = input.description.unwrap_or_default();
        let model = input.model.unwrap_or_else(|| context.model.clone());
        let system = input.system.unwrap_or_else(|| context.system.clone());

        let mut agent = Agent::new(
            context.provider.clone(),
            context.tools.clone(),
            model,
            system,
        );
        agent.max_tokens = context.max_tokens;
        agent.max_iterations = context.max_iterations;
        agent.depth = context.depth + 1;
        agent.max_depth = context.max_depth;

        let agent_id = agent.session_id.clone();
        info!(
            agent_id,
            description,
            run_in_background = input.run_in_background,
            "Spawning sub-agent"
        );

        if input.run_in_background {
            let background_tx = context.background_tx.clone();
            let pending = context.pending_background.clone();
            let bg_agent_id = agent_id.clone();
            let bg_description = description.clone();
            let prompt = input.prompt;

            pending.fetch_add(1, Ordering::SeqCst);

            let bg_handle = tokio::spawn(async move {
                let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(128);
                let start = Instant::now();

                let handle = tokio::spawn(async move { agent.run(prompt, event_tx).await });
                // If this outer task is aborted, also abort the inner agent task.
                let _guard = AbortOnDrop(handle.abort_handle());

                let (output, total_tokens, tool_use_count, had_error) =
                    collect_agent_output(&mut event_rx).await;

                let duration_ms = start.elapsed().as_millis() as u64;

                let (result, is_error) = match handle.await {
                    Ok(Ok(())) if !had_error => {
                        let text = if output.is_empty() {
                            "(sub-agent produced no text output)".to_string()
                        } else {
                            output
                        };
                        (text, false)
                    }
                    Ok(Ok(())) => (output, true),
                    Ok(Err(e)) => (format!("Sub-agent failed: {e}"), true),
                    Err(e) => (format!("Sub-agent panicked: {e}"), true),
                };

                let _ = background_tx
                    .send(BackgroundAgentResult {
                        agent_id: bg_agent_id,
                        description: bg_description,
                        result,
                        is_error,
                        duration_ms,
                        total_tokens,
                        tool_use_count,
                    })
                    .await;

                // Decrement after send so the parent's wait loop sees pending > 0
                // until the result is in the channel and ready to be received.
                pending.fetch_sub(1, Ordering::SeqCst);
            });

            context.background_handles.lock().await.push(bg_handle);

            ToolResult::success(format!(
                "Agent started in background.\nagentId: {agent_id}\ndescription: {description}"
            ))
        } else {
            let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(128);
            let start = Instant::now();

            let prompt = input.prompt;
            let handle = tokio::spawn(async move { agent.run(prompt, event_tx).await });
            // Abort inner agent if this tool call is cancelled (e.g. parent aborted).
            let _guard = AbortOnDrop(handle.abort_handle());

            let (output, total_tokens, tool_use_count, had_error) =
                collect_agent_output(&mut event_rx).await;

            let duration_ms = start.elapsed().as_millis() as u64;

            if had_error {
                handle.abort();
                return ToolResult::error(format!("Sub-agent error: {output}"));
            }

            match handle.await {
                Ok(Ok(())) => {
                    let text = if output.is_empty() {
                        "(sub-agent produced no text output)".to_string()
                    } else {
                        output
                    };
                    let result = format!(
                        "{text}\nagentId: {agent_id} (use SendMessage with to: '{agent_id}' to continue this agent)\n<usage>\ntotal_tokens: {total_tokens}\ntool_uses: {tool_use_count}\nduration_ms: {duration_ms}\n</usage>"
                    );
                    ToolResult::success(result)
                }
                Ok(Err(e)) => ToolResult::error(format!("Sub-agent failed: {e}")),
                Err(e) => ToolResult::error(format!("Sub-agent panicked: {e}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, ToolResultContent};
    use crate::provider::{Chunk, Provider, Request, StopReason};
    use crate::tools::ToolRegistry;
    use std::sync::Arc;

    struct EchoProvider;

    #[async_trait]
    impl Provider for EchoProvider {
        async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
            // Just echo back the user's prompt.
            let text = req
                .messages
                .first()
                .and_then(|m| m.content.first())
                .map(|c| match c {
                    ContentBlock::Text { text } => text.clone(),
                    _ => "no text".into(),
                })
                .unwrap_or_default();

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

    fn make_context(
        provider: Arc<dyn Provider>,
    ) -> (ToolContext, mpsc::Receiver<super::BackgroundAgentResult>) {
        let (background_tx, background_rx) = mpsc::channel(16);
        let ctx = ToolContext {
            provider,
            tools: Arc::new(ToolRegistry::new()),
            model: "test".into(),
            system: "test system".into(),
            max_tokens: 1024,
            max_iterations: 100,
            depth: 0,
            max_depth: 3,
            background_tx,
            pending_background: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            background_handles: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        };
        (ctx, background_rx)
    }

    #[tokio::test]
    async fn sub_agent_returns_text() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let (ctx, _bg_rx) = make_context(provider);

        let tool = AgentTool;
        let result = tool
            .call(json!({"prompt": "hello from sub-agent"}), &ctx)
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("hello from sub-agent"));
        assert!(text.contains("<usage>"));
        assert!(text.contains("total_tokens:"));
        assert!(text.contains("duration_ms:"));
    }

    #[tokio::test]
    async fn max_depth_rejected() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let (mut ctx, _bg_rx) = make_context(provider);
        ctx.depth = 3;
        ctx.max_depth = 3;

        let tool = AgentTool;
        let result = tool.call(json!({"prompt": "deep"}), &ctx).await;
        assert!(result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("Max agent depth"));
    }

    #[tokio::test]
    async fn custom_system_prompt() {
        // Use a provider that echoes the system prompt to verify it's passed through.
        struct SystemEchoProvider;

        #[async_trait]
        impl Provider for SystemEchoProvider {
            async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let _ = tx.send(Chunk::Text(req.system.clone())).await;
                let _ = tx
                    .send(Chunk::Done {
                        stop_reason: StopReason::EndTurn,
                        content: vec![ContentBlock::Text { text: req.system }],
                    })
                    .await;
                Ok(())
            }
        }

        let provider: Arc<dyn Provider> = Arc::new(SystemEchoProvider);
        let (ctx, _bg_rx) = make_context(provider);

        let tool = AgentTool;
        let result = tool
            .call(json!({"prompt": "test", "system": "You are a cat."}), &ctx)
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("You are a cat."));
    }

    #[tokio::test]
    async fn model_override() {
        // Use a provider that echoes the model name.
        struct ModelEchoProvider;

        #[async_trait]
        impl Provider for ModelEchoProvider {
            async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let _ = tx.send(Chunk::Text(req.model.clone())).await;
                let _ = tx
                    .send(Chunk::Done {
                        stop_reason: StopReason::EndTurn,
                        content: vec![ContentBlock::Text { text: req.model }],
                    })
                    .await;
                Ok(())
            }
        }

        let provider: Arc<dyn Provider> = Arc::new(ModelEchoProvider);
        let (ctx, _bg_rx) = make_context(provider);

        let tool = AgentTool;
        let result = tool
            .call(
                json!({"prompt": "test", "model": "claude-haiku-4-5-20251001"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("claude-haiku-4-5-20251001"));
    }

    #[tokio::test]
    async fn tool_use_counted() {
        // Provider that does one tool call then responds.
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
                    let input = json!({"command": "echo hi"});
                    let _ = tx
                        .send(Chunk::ToolUse {
                            id: "t1".into(),
                            name: "bash".into(),
                            input: input.clone(),
                        })
                        .await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::ToolUse,
                            content: vec![ContentBlock::ToolUse {
                                id: "t1".into(),
                                name: "bash".into(),
                                input,
                            }],
                        })
                        .await;
                }
                Ok(())
            }
        }

        let provider: Arc<dyn Provider> = Arc::new(ToolCallProvider);
        let (mut ctx, _bg_rx) = make_context(provider);
        ctx.tools = Arc::new(ToolRegistry::with_defaults());

        let tool = AgentTool;
        let result = tool.call(json!({"prompt": "run echo"}), &ctx).await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(
            text.contains("tool_uses: 1"),
            "should count 1 tool use, got: {text}"
        );
    }

    #[tokio::test]
    async fn sub_agent_provider_error() {
        struct FailProvider;

        #[async_trait]
        impl Provider for FailProvider {
            async fn request(&self, _req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let _ = tx.send(Chunk::Error("provider broke".into())).await;
                anyhow::bail!("provider broke");
            }
        }

        let provider: Arc<dyn Provider> = Arc::new(FailProvider);
        let (ctx, _bg_rx) = make_context(provider);

        let tool = AgentTool;
        let result = tool.call(json!({"prompt": "test"}), &ctx).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn missing_prompt_rejected() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let (ctx, _bg_rx) = make_context(provider);

        let tool = AgentTool;
        let result = tool.call(json!({}), &ctx).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn run_in_background_returns_immediately() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let (ctx, mut bg_rx) = make_context(provider);

        let tool = AgentTool;
        let result = tool
            .call(
                json!({"prompt": "background task", "run_in_background": true, "description": "test bg"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("Agent started in background"));
        assert!(text.contains("agentId:"));

        // Wait for background agent to complete and send result.
        let bg_result = tokio::time::timeout(std::time::Duration::from_secs(5), bg_rx.recv())
            .await
            .expect("background agent should complete within timeout")
            .expect("should receive background result");

        assert!(!bg_result.is_error);
        assert_eq!(bg_result.description, "test bg");
        assert!(bg_result.result.contains("background task"));
    }

    #[tokio::test]
    async fn run_in_background_error_reported() {
        struct FailProvider;

        #[async_trait]
        impl Provider for FailProvider {
            async fn request(&self, _req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let _ = tx.send(Chunk::Error("bg provider broke".into())).await;
                anyhow::bail!("bg provider broke");
            }
        }

        let provider: Arc<dyn Provider> = Arc::new(FailProvider);
        let (ctx, mut bg_rx) = make_context(provider);

        let tool = AgentTool;
        let result = tool
            .call(json!({"prompt": "fail", "run_in_background": true}), &ctx)
            .await;
        assert!(!result.is_error, "should return success (agent started)");

        let bg_result = tokio::time::timeout(std::time::Duration::from_secs(5), bg_rx.recv())
            .await
            .expect("should complete within timeout")
            .expect("should receive background result");

        assert!(bg_result.is_error);
    }

    #[tokio::test]
    async fn run_in_background_respects_max_depth() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let (mut ctx, _bg_rx) = make_context(provider);
        ctx.depth = 3;
        ctx.max_depth = 3;

        let tool = AgentTool;
        let result = tool
            .call(
                json!({"prompt": "deep bg", "run_in_background": true}),
                &ctx,
            )
            .await;
        assert!(result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("Max agent depth"));
    }
}
