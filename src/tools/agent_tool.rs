use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::message::{StreamEvent, ToolResult};

use super::{Tool, ToolContext};

pub struct AgentTool;

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default)]
    system: Option<String>,
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
                "system": {
                    "type": "string",
                    "description": "Optional system prompt override for the sub-agent"
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

        let system = input.system.unwrap_or_else(|| context.system.clone());

        let mut agent = Agent::new(
            context.provider.clone(),
            context.tools.clone(),
            context.model.clone(),
            system,
        );
        agent.max_tokens = context.max_tokens;
        agent.max_iterations = context.max_iterations;
        agent.depth = context.depth + 1;
        agent.max_depth = context.max_depth;

        let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(128);

        let prompt = input.prompt;
        let handle = tokio::spawn(async move { agent.run(prompt, event_tx).await });

        // Collect the sub-agent's text output from assistant messages.
        let mut output = String::new();
        while let Some(event) = event_rx.recv().await {
            match event {
                StreamEvent::Assistant { ref message, .. } => {
                    for block in &message.content {
                        if let crate::message::ContentBlock::Text { text } = block {
                            output.push_str(text);
                        }
                    }
                }
                StreamEvent::Result {
                    is_error: true,
                    subtype,
                    ..
                } => {
                    handle.abort();
                    return ToolResult::error(format!("Sub-agent error: {subtype}"));
                }
                _ => {}
            }
        }

        match handle.await {
            Ok(Ok(())) => {
                if output.is_empty() {
                    ToolResult::success("(sub-agent produced no text output)")
                } else {
                    ToolResult::success(output)
                }
            }
            Ok(Err(e)) => ToolResult::error(format!("Sub-agent failed: {e}")),
            Err(e) => ToolResult::error(format!("Sub-agent panicked: {e}")),
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

    fn make_context(provider: Arc<dyn Provider>) -> ToolContext {
        ToolContext {
            provider,
            tools: Arc::new(ToolRegistry::new()),
            model: "test".into(),
            system: "test system".into(),
            max_tokens: 1024,
            max_iterations: 100,
            depth: 0,
            max_depth: 3,
        }
    }

    #[tokio::test]
    async fn sub_agent_returns_text() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        let tool = AgentTool;
        let result = tool
            .call(json!({"prompt": "hello from sub-agent"}), &ctx)
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("hello from sub-agent"));
    }

    #[tokio::test]
    async fn max_depth_rejected() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let mut ctx = make_context(provider);
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
        let ctx = make_context(provider);

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
        let ctx = make_context(provider);

        let tool = AgentTool;
        let result = tool.call(json!({"prompt": "test"}), &ctx).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn missing_prompt_rejected() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        let tool = AgentTool;
        let result = tool.call(json!({}), &ctx).await;
        assert!(result.is_error);
    }
}
