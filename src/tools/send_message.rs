use std::time::Instant;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::info;

use crate::agent::Agent;
use crate::message::{StreamEvent, ToolResult};

use super::{Tool, ToolContext, collect_agent_output};

pub struct SendMessageTool;

#[derive(Deserialize)]
struct Input {
    to: String,
    message: String,
}

#[async_trait]
impl Tool for SendMessageTool {
    fn name(&self) -> &str {
        "send_message"
    }

    fn description(&self) -> &str {
        "Send a message to a previously spawned agent. If the agent is currently running \
         in the background, the message is queued for delivery at its next tool round. \
         If the agent has completed, it is resumed with its full context preserved. \
         Use the agent's name or ID as the 'to' field."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "The agent's name or ID to send the message to"
                },
                "message": {
                    "type": "string",
                    "description": "The message to send to the agent"
                }
            },
            "required": ["to", "message"]
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

        // Resolve agent ID and queue if running — hold running_agents lock through
        // the entire check+queue to prevent race with background agent deregister+drain.
        {
            let running = context.running_agents.lock().await;
            let resolved_id = if running.contains(&input.to) {
                Some(input.to.clone())
            } else {
                // Try name registry for running background agents.
                context
                    .agent_name_registry
                    .lock()
                    .await
                    .get(&input.to)
                    .cloned()
            };

            if let Some(ref agent_id) = resolved_id
                && running.contains(agent_id)
            {
                info!(
                    agent_id,
                    to = input.to,
                    "Queueing message for running agent"
                );
                context
                    .pending_messages
                    .lock()
                    .await
                    .entry(agent_id.clone())
                    .or_default()
                    .push(input.message);
                return ToolResult::success(format!(
                    "Message queued for delivery to '{}' at its next tool round.\nagentId: {agent_id}",
                    input.to
                ));
            }
        }

        // Agent is not running — try to resume from completed agents.
        let mut agents = context.completed_agents.lock().await;
        let agent_key = {
            // Try direct ID lookup first.
            if agents.contains_key(&input.to) {
                Some(input.to.clone())
            } else {
                // Search by name.
                agents
                    .iter()
                    .find(|(_, a)| a.name.as_deref() == Some(&input.to))
                    .map(|(k, _)| k.clone())
            }
        };

        let Some(key) = agent_key else {
            return ToolResult::error(format!(
                "Agent '{}' not found. It may not exist or may still be starting up.",
                input.to
            ));
        };

        // Remove the agent from the map — we'll re-insert after completion.
        let completed = agents.remove(&key).unwrap();
        drop(agents); // Release lock before running.

        let agent_id = completed.agent_id.clone();
        let agent_name = completed.name.clone();
        info!(agent_id, to = input.to, "Resuming agent via SendMessage");

        let mut agent = Agent::new(
            context.provider.clone(),
            completed.tools.clone(),
            completed.model.clone(),
            completed.system.clone(),
        );
        agent.max_tokens = context.max_tokens;
        agent.max_iterations = context.max_iterations;
        agent.depth = context.depth + 1;
        agent.max_depth = context.max_depth;
        agent.cwd = completed.cwd.clone();
        // Keep the same session_id for continuity.
        agent.session_id = agent_id.clone();

        let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(128);
        let start = Instant::now();

        let prev_messages = completed.messages.clone();
        let model = completed.model.clone();
        let system = completed.system.clone();
        let tools = completed.tools.clone();
        let cwd = completed.cwd.clone();
        let msg = input.message;

        // Helper to re-insert the agent with original messages on failure.
        let restore_agent = |msgs: Vec<crate::message::Message>| super::CompletedAgent {
            agent_id: agent_id.clone(),
            name: agent_name.clone(),
            messages: msgs,
            model: model.clone(),
            system: system.clone(),
            tools: tools.clone(),
            cwd: cwd.clone(),
        };

        let handle = tokio::spawn(async move { agent.resume(prev_messages, msg, event_tx).await });
        // Abort inner agent if this tool call is cancelled (e.g. parent aborted).
        let _guard = super::agent_tool::AbortOnDrop(handle.abort_handle());

        let (output, total_tokens, tool_use_count, had_error) =
            collect_agent_output(&mut event_rx).await;
        let duration_ms = start.elapsed().as_millis() as u64;

        if had_error {
            handle.abort();
            // Restore agent so it can be retried.
            context
                .completed_agents
                .lock()
                .await
                .insert(agent_id.clone(), restore_agent(completed.messages));
            return ToolResult::error(format!("Agent error: {output}"));
        }

        let (result_text, new_messages) = match handle.await {
            Ok(Ok(msgs)) => {
                let text = if output.is_empty() {
                    "(agent produced no text output)".to_string()
                } else {
                    output
                };
                (Ok(text), Some(msgs))
            }
            Ok(Err(e)) => (Err(format!("Agent failed: {e}")), None),
            Err(e) => (Err(format!("Agent panicked: {e}")), None),
        };

        match result_text {
            Ok(text) => {
                // Store updated agent state for further resumption.
                if let Some(msgs) = new_messages {
                    context
                        .completed_agents
                        .lock()
                        .await
                        .insert(agent_id.clone(), restore_agent(msgs));
                }

                let name_info = agent_name
                    .as_deref()
                    .map(|n| format!("\nagentName: {n}"))
                    .unwrap_or_default();
                let result = format!(
                    "{text}\nagentId: {agent_id}{name_info}\n<usage>\ntotal_tokens: {total_tokens}\ntool_uses: {tool_use_count}\nduration_ms: {duration_ms}\n</usage>"
                );
                ToolResult::success(result)
            }
            Err(e) => {
                // Restore agent with original messages so it can be retried.
                context
                    .completed_agents
                    .lock()
                    .await
                    .insert(agent_id.clone(), restore_agent(completed.messages));
                ToolResult::error(e)
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
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    struct EchoProvider;

    #[async_trait]
    impl Provider for EchoProvider {
        async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
            let text = req
                .messages
                .last()
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
        let (background_tx, _) = mpsc::channel(16);
        ToolContext {
            provider,
            tools: Arc::new(ToolRegistry::with_defaults()),
            model: "test".into(),
            system: "test system".into(),
            max_tokens: 1024,
            max_iterations: 100,
            depth: 0,
            max_depth: 3,
            background_tx,
            pending_background: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            background_handles: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            background_worktrees: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            completed_agents: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            running_agents: Arc::new(tokio::sync::Mutex::new(HashSet::new())),
            agent_name_registry: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            pending_messages: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            parent_messages: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            cwd: None,
            skills: Arc::new(crate::skill::SkillRegistry::new()),
        }
    }

    #[tokio::test]
    async fn agent_not_found() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        let tool = SendMessageTool;
        let result = tool
            .call(json!({"to": "nonexistent", "message": "hi"}), &ctx)
            .await;
        assert!(result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("not found"));
    }

    #[tokio::test]
    async fn resume_agent_by_id() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        // First, spawn an agent to create a completed entry.
        let agent_tool = super::super::AgentTool;
        let result = agent_tool
            .call(json!({"prompt": "initial task"}), &ctx)
            .await;
        assert!(!result.is_error);

        // Extract agent ID from result.
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        let agent_id = text
            .lines()
            .find(|l| l.starts_with("agentId: "))
            .unwrap()
            .strip_prefix("agentId: ")
            .unwrap()
            .trim();

        // Verify agent is in completed_agents.
        assert!(ctx.completed_agents.lock().await.contains_key(agent_id));

        // Now send a message to resume it.
        let tool = SendMessageTool;
        let result = tool
            .call(
                json!({"to": agent_id, "message": "follow-up question"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("follow-up question"));
        assert!(text.contains("agentId:"));
    }

    #[tokio::test]
    async fn resume_agent_by_name() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        // Spawn a named agent.
        let agent_tool = super::super::AgentTool;
        let result = agent_tool
            .call(
                json!({"prompt": "initial task", "name": "my-researcher"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("agentName: my-researcher"));

        // Resume by name.
        let tool = SendMessageTool;
        let result = tool
            .call(
                json!({"to": "my-researcher", "message": "continue research"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("continue research"));
    }

    #[tokio::test]
    async fn multiple_resumptions() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        // Spawn agent.
        let agent_tool = super::super::AgentTool;
        let result = agent_tool
            .call(json!({"prompt": "task 1", "name": "worker"}), &ctx)
            .await;
        assert!(!result.is_error);

        // Resume twice.
        let tool = SendMessageTool;
        let result = tool
            .call(json!({"to": "worker", "message": "task 2"}), &ctx)
            .await;
        assert!(!result.is_error);

        let result = tool
            .call(json!({"to": "worker", "message": "task 3"}), &ctx)
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("task 3"));
    }

    #[tokio::test]
    async fn missing_required_fields() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        let tool = SendMessageTool;
        let result = tool.call(json!({"to": "agent1"}), &ctx).await;
        assert!(result.is_error);

        let result = tool.call(json!({"message": "hi"}), &ctx).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn max_depth_rejected() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let mut ctx = make_context(provider);
        ctx.depth = 3;
        ctx.max_depth = 3;

        let tool = SendMessageTool;
        let result = tool.call(json!({"to": "any", "message": "hi"}), &ctx).await;
        assert!(result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("Max agent depth"));
    }

    #[tokio::test]
    async fn duplicate_name_resumes_first_found() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        // Spawn two agents with the same name.
        let agent_tool = super::super::AgentTool;
        let result1 = agent_tool
            .call(json!({"prompt": "agent one", "name": "dup"}), &ctx)
            .await;
        assert!(!result1.is_error);

        let result2 = agent_tool
            .call(json!({"prompt": "agent two", "name": "dup"}), &ctx)
            .await;
        assert!(!result2.is_error);

        // Both should be in the map (different IDs).
        let agents = ctx.completed_agents.lock().await;
        let dup_count = agents
            .values()
            .filter(|a| a.name.as_deref() == Some("dup"))
            .count();
        assert_eq!(dup_count, 2);
        drop(agents);

        // SendMessage by name resolves to one of them (not an error).
        let tool = SendMessageTool;
        let result = tool
            .call(json!({"to": "dup", "message": "which one?"}), &ctx)
            .await;
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn queue_message_for_running_agent() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        let agent_id = "running-agent-123".to_string();
        // Simulate a running background agent.
        ctx.running_agents.lock().await.insert(agent_id.clone());

        let tool = SendMessageTool;
        let result = tool
            .call(
                json!({"to": &agent_id, "message": "hello running agent"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("queued"));
        assert!(text.contains(&agent_id));

        // Verify the message was queued.
        let pm = ctx.pending_messages.lock().await;
        let msgs = pm.get(&agent_id).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0], "hello running agent");
    }

    #[tokio::test]
    async fn queue_message_by_name_for_running_agent() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        let agent_id = "running-named-456".to_string();
        ctx.running_agents.lock().await.insert(agent_id.clone());
        ctx.agent_name_registry
            .lock()
            .await
            .insert("my-worker".into(), agent_id.clone());

        let tool = SendMessageTool;
        let result = tool
            .call(json!({"to": "my-worker", "message": "task update"}), &ctx)
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("queued"));

        let pm = ctx.pending_messages.lock().await;
        assert_eq!(pm.get(&agent_id).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn stale_name_registry_falls_through_to_completed() {
        // name_registry has an entry but the agent is NOT in running_agents.
        // This simulates a stale registry entry. SendMessage should NOT queue;
        // it should fall through and resume the completed agent instead.
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        // Create a completed agent with a name.
        let agent_tool = super::super::AgentTool;
        let result = agent_tool
            .call(json!({"prompt": "task", "name": "stale-worker"}), &ctx)
            .await;
        assert!(!result.is_error);

        // Add a stale name registry entry (agent is completed, not running).
        let agent_id = {
            let agents = ctx.completed_agents.lock().await;
            agents
                .iter()
                .find(|(_, a)| a.name.as_deref() == Some("stale-worker"))
                .map(|(k, _)| k.clone())
                .unwrap()
        };
        ctx.agent_name_registry
            .lock()
            .await
            .insert("stale-worker".into(), agent_id.clone());
        // Note: agent_id is NOT in running_agents.

        // SendMessage should resume (not queue).
        let tool = SendMessageTool;
        let result = tool
            .call(json!({"to": "stale-worker", "message": "follow up"}), &ctx)
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        // Should be a resume result, not a "queued" result.
        assert!(!text.contains("queued"));
        assert!(text.contains("agentId:"));

        // pending_messages should be empty — nothing was queued.
        let pm = ctx.pending_messages.lock().await;
        assert!(pm.get(&agent_id).is_none());
    }

    #[tokio::test]
    async fn multiple_queued_messages() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let ctx = make_context(provider);

        let agent_id = "running-multi-789".to_string();
        ctx.running_agents.lock().await.insert(agent_id.clone());

        let tool = SendMessageTool;
        tool.call(json!({"to": &agent_id, "message": "msg1"}), &ctx)
            .await;
        tool.call(json!({"to": &agent_id, "message": "msg2"}), &ctx)
            .await;
        tool.call(json!({"to": &agent_id, "message": "msg3"}), &ctx)
            .await;

        let pm = ctx.pending_messages.lock().await;
        let msgs = pm.get(&agent_id).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0], "msg1");
        assert_eq!(msgs[1], "msg2");
        assert_eq!(msgs[2], "msg3");
    }
}
