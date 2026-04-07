use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::info;

use crate::agent::Agent;
use crate::message::{StreamEvent, ToolResult};

use super::{BackgroundAgentResult, Tool, ToolContext, WorktreeInfo, git_cmd};

pub struct AgentTool;

/// Aborts the associated task when dropped.
/// Ensures inner spawned tasks are cleaned up if the outer task is cancelled.
pub(crate) struct AbortOnDrop(pub(crate) tokio::task::AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Worktree state for cleanup after sub-agent completes.
struct Worktree {
    path: String,
    branch: String,
    start_commit: String,
    git_root: String,
}

impl Worktree {
    /// Create a git worktree. `cwd` is the directory to run git commands from
    /// (used to find the git repo). If None, uses the process CWD.
    async fn create(cwd: Option<&str>) -> Result<Self, String> {
        let id = crate::message::new_uuid();
        let short_id = &id[..8];
        let branch = format!("worktree-{short_id}");

        // Find git root.
        let mut cmd = git_cmd();
        cmd.args(["rev-parse", "--show-toplevel"]);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let output = cmd
            .output()
            .await
            .map_err(|e| format!("Failed to find git root: {e}"))?;
        if !output.status.success() {
            return Err("Not in a git repository".into());
        }
        let git_root = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let worktree_path = format!("{git_root}/.coo/worktrees/{short_id}");

        // Create worktree from HEAD.
        let mut cmd = git_cmd();
        cmd.args(["worktree", "add", &worktree_path, "-b", &branch, "HEAD"]);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let output = cmd
            .output()
            .await
            .map_err(|e| format!("Failed to create worktree: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git worktree add failed: {stderr}"));
        }

        // Record starting commit from the worktree itself (not the main repo),
        // to avoid a race where HEAD advances between rev-parse and worktree add.
        let start_commit = match git_cmd()
            .args(["rev-parse", "HEAD"])
            .current_dir(&worktree_path)
            .output()
            .await
        {
            Ok(output) => String::from_utf8_lossy(&output.stdout).trim().to_string(),
            Err(e) => {
                // Worktree was created but we can't read HEAD — clean up.
                let _ = git_cmd()
                    .args(["worktree", "remove", "--force", &worktree_path])
                    .current_dir(&git_root)
                    .output()
                    .await;
                let _ = git_cmd()
                    .args(["branch", "-D", &branch])
                    .current_dir(&git_root)
                    .output()
                    .await;
                return Err(format!("Failed to get worktree HEAD: {e}"));
            }
        };

        Ok(Self {
            path: worktree_path,
            branch,
            start_commit,
            git_root,
        })
    }

    /// Check if the worktree has uncommitted changes or new commits.
    async fn has_changes(&self) -> bool {
        // Check uncommitted changes.
        let has_uncommitted = git_cmd()
            .args(["status", "--porcelain"])
            .current_dir(&self.path)
            .output()
            .await
            .is_ok_and(|o| !o.stdout.is_empty());
        if has_uncommitted {
            return true;
        }

        // Check if HEAD advanced beyond the starting commit.
        git_cmd()
            .args(["rev-parse", "HEAD"])
            .current_dir(&self.path)
            .output()
            .await
            .is_ok_and(|o| {
                let current = String::from_utf8_lossy(&o.stdout).trim().to_string();
                current != self.start_commit
            })
    }

    /// Remove the worktree and delete the branch.
    async fn cleanup(&self) {
        let _ = git_cmd()
            .args(["worktree", "remove", "--force", &self.path])
            .current_dir(&self.git_root)
            .output()
            .await;
        let _ = git_cmd()
            .args(["branch", "-D", &self.branch])
            .current_dir(&self.git_root)
            .output()
            .await;
    }
}

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    run_in_background: bool,
    #[serde(default)]
    isolation: Option<String>,
    #[serde(default)]
    subagent_type: Option<String>,
}

use super::collect_agent_output;

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        "agent"
    }

    fn description(&self) -> &str {
        "Launch a sub-agent to handle complex, multi-step tasks autonomously.\n\n\
         Each agent runs in its own context with its own tool access. \
         Each invocation starts fresh — provide a complete task description.\n\n\
         ## Available agent types\n\n\
         - general-purpose (default): Full tool access. For complex multi-step tasks, \
         code changes, and work requiring both research and modification. \
         (Tools: all)\n\
         - explore: Fast codebase exploration and search. Cannot modify files. \
         Use for finding files, searching code, and answering questions about the codebase. \
         (Tools: bash, glob, grep, read, web_fetch)\n\
         - plan: Software architecture research. Cannot modify files. \
         Use for designing implementation plans, identifying critical files, \
         and considering trade-offs. \
         (Tools: bash, glob, grep, read, web_fetch)\n\n\
         ## When NOT to use the Agent tool\n\n\
         - To read a specific file: use read directly\n\
         - To search for a specific pattern: use grep or glob directly\n\
         - For simple single-step tasks: use the tool directly\n\n\
         ## Usage notes\n\n\
         - Always include a short description (3-5 words) summarizing the task.\n\
         - Launch multiple agents concurrently when tasks are independent. \
         To launch in parallel, make multiple agent tool calls in a single message.\n\
         - Foreground vs background: Use foreground (default) when you need the agent's \
         results before you can proceed — e.g., research whose findings inform your next steps. \
         Use background when you have genuinely independent work to do in parallel.\n\
         - When an agent runs in the background, you will be automatically notified when \
         it completes — do NOT sleep, poll, or proactively check on its progress. \
         Continue with other work or respond to the user instead.\n\
         - The agent result is not visible to the user. Summarize it in your response.\n\
         - The agent's outputs should generally be trusted.\n\
         - Clearly tell the agent whether to write code or just research.\n\
         - Set isolation: 'worktree' for agents that modify files in parallel.\n\
         - To continue a previously spawned agent, use send_message with the \
         agent's name or ID. The agent resumes with its full context preserved.\n\n\
         ## Writing the prompt\n\n\
         Brief the agent like a colleague who just walked in — no prior context.\n\
         - Explain what you want and why.\n\
         - Describe what you've already tried or ruled out.\n\
         - Give enough context for the agent to make judgment calls.\n\
         - Never delegate understanding: don't write 'based on your findings, fix it'. \
         Include file paths, line numbers, and what specifically to change.\n\
         - Terse command-style prompts produce shallow, generic work.\n\n\
         ## Example\n\n\
         prompt: \"Find all usages of the deprecated parse_config() function in src/ \
         and report which files and line numbers still call it.\"\n\
         description: \"find deprecated parse_config usage\"\n\
         subagent_type: \"explore\""
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the sub-agent to perform. Be specific about what you want, including constraints and context. The agent has no prior conversation history."
                },
                "description": {
                    "type": "string",
                    "description": "A short (3-5 word) description of the task"
                },
                "name": {
                    "type": "string",
                    "description": "Optional name for the agent. Used to reference the agent later via SendMessage. If omitted, the agent can still be referenced by its ID."
                },
                "model": {
                    "type": "string",
                    "enum": ["sonnet", "opus", "haiku"],
                    "description": "Optional model override for this agent. Takes precedence over the subagent_type's model. If omitted, inherits from the parent agent."
                },
                "system": {
                    "type": "string",
                    "description": "Optional system prompt override for the sub-agent"
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Run the agent in the background. Returns immediately with agent ID. You will be automatically notified when it completes — do NOT sleep or poll."
                },
                "isolation": {
                    "type": "string",
                    "enum": ["worktree"],
                    "description": "Set to 'worktree' to run in an isolated git worktree. The worktree is cleaned up if no changes are made; otherwise the path and branch are returned."
                },
                "subagent_type": {
                    "type": "string",
                    "description": "The type of agent to use for this task"
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

        if let Some(ref iso) = input.isolation
            && iso != "worktree"
        {
            return ToolResult::error(format!(
                "Unknown isolation mode: {iso}. Supported: 'worktree'"
            ));
        }

        // Look up sub-agent type preset (if specified).
        // "general-purpose" is the default — treat it the same as omitting subagent_type.
        let agent_type = match &input.subagent_type {
            Some(name) if name == "general-purpose" => None,
            Some(name) => match super::get_subagent_type(name) {
                Some(t) => Some(t),
                None => {
                    let known: Vec<&str> = super::SUBAGENT_TYPES.iter().map(|t| t.name).collect();
                    return ToolResult::error(format!(
                        "Unknown subagent_type: {name}. Available: general-purpose, {known:?}"
                    ));
                }
            },
            None => None,
        };

        let description = input.description.unwrap_or_default();
        // subagent_type defaults < explicit input overrides
        // Friendly names (sonnet/opus/haiku) are resolved to full model IDs.
        let model = input
            .model
            .map(|m| super::resolve_model_name(&m).to_string())
            .unwrap_or_else(|| {
                agent_type
                    .and_then(|t| t.model)
                    .unwrap_or(&context.model)
                    .to_string()
            });
        let system = input.system.unwrap_or_else(|| {
            agent_type
                .and_then(|t| t.system)
                .unwrap_or(&context.system)
                .to_string()
        });
        let use_worktree = input.isolation.as_deref() == Some("worktree");

        // Set up worktree if requested.
        let worktree = if use_worktree {
            match Worktree::create(context.cwd.as_deref()).await {
                Ok(wt) => Some(wt),
                Err(e) => return ToolResult::error(format!("Failed to create worktree: {e}")),
            }
        } else {
            None
        };

        // Apply tool filtering for sub-agent types with denied_tools.
        let tools = if let Some(t) = agent_type
            && !t.denied_tools.is_empty()
        {
            Arc::new(
                context
                    .tools
                    .filtered(|name| !t.denied_tools.contains(&name)),
            )
        } else {
            context.tools.clone()
        };

        let agent_name = input.name.clone();
        let mut agent = Agent::new(
            context.provider.clone(),
            tools.clone(),
            model.clone(),
            system.clone(),
        );
        agent.max_tokens = context.max_tokens;
        agent.max_iterations = context.max_iterations;
        agent.depth = context.depth + 1;
        agent.max_depth = context.max_depth;
        agent.cwd = worktree.as_ref().map(|wt| wt.path.clone());
        // Fork optimization: inherit parent messages as prefix for prompt cache reuse.
        agent.prefix_messages = context.parent_messages.lock().await.clone();
        let agent_cwd = agent.cwd.clone();

        let agent_id = agent.session_id.clone();
        let subagent_type_name = agent_type.map(|t| t.name).unwrap_or("general-purpose");
        info!(
            agent_id,
            description,
            subagent_type = subagent_type_name,
            run_in_background = input.run_in_background,
            use_worktree,
            "Spawning sub-agent"
        );

        if input.run_in_background {
            // Register worktree for cleanup in case the bg task is aborted.
            // Double cleanup is harmless (git worktree remove on non-existent is a no-op).
            if let Some(ref wt) = worktree {
                context
                    .background_worktrees
                    .lock()
                    .await
                    .push(WorktreeInfo {
                        path: wt.path.clone(),
                        branch: wt.branch.clone(),
                        git_root: wt.git_root.clone(),
                    });
            }

            let background_tx = context.background_tx.clone();
            let pending = context.pending_background.clone();
            let bg_worktrees = context.background_worktrees.clone();
            let bg_agent_id = agent_id.clone();
            let bg_description = description.clone();
            let prompt = input.prompt;
            let wt_path_for_removal = worktree.as_ref().map(|wt| wt.path.clone());

            pending.fetch_add(1, Ordering::SeqCst);

            let bg_handle = tokio::spawn(async move {
                // Guard ensures pending is decremented even on panic/abort.
                // Dropped after send, so the parent's wait loop sees pending > 0
                // until the result is in the channel and ready to be received.
                let _pending_guard = super::PendingGuard(pending);

                let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(128);
                let start = Instant::now();

                let handle = tokio::spawn(async move { agent.run(prompt, event_tx).await });
                // If this outer task is aborted, also abort the inner agent task.
                let _guard = AbortOnDrop(handle.abort_handle());

                let (output, total_tokens, tool_use_count, had_error) =
                    collect_agent_output(&mut event_rx).await;

                let duration_ms = start.elapsed().as_millis() as u64;

                let (mut result, is_error, _agent_messages) = match handle.await {
                    Ok(Ok(msgs)) if !had_error => {
                        let text = if output.is_empty() {
                            "(sub-agent produced no text output)".to_string()
                        } else {
                            output
                        };
                        (text, false, Some(msgs))
                    }
                    Ok(Ok(_)) => (output, true, None),
                    Ok(Err(e)) => (format!("Sub-agent failed: {e}"), true, None),
                    Err(e) => (format!("Sub-agent panicked: {e}"), true, None),
                };

                // Handle worktree cleanup.
                if let Some(wt) = worktree {
                    if is_error || !wt.has_changes().await {
                        wt.cleanup().await;
                    } else {
                        result.push_str(&format!(
                            "\nworktree_path: {}\nworktree_branch: {}",
                            wt.path, wt.branch
                        ));
                    }
                }

                // Remove from background_worktrees now that the task has handled cleanup.
                // This prevents the parent's abort path from cleaning up worktrees that
                // were intentionally kept (has changes) by a completed agent.
                if let Some(ref path) = wt_path_for_removal {
                    bg_worktrees.lock().await.retain(|w| w.path != *path);
                }

                // Note: background agents don't store in completed_agents for now,
                // since they lack access to the agent's config after the spawn.
                // This could be added later if needed.

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

                // Drop _pending_guard here (end of block) to decrement after send.
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
                if let Some(wt) = &worktree {
                    wt.cleanup().await;
                }
                return ToolResult::error(format!("Sub-agent error: {output}"));
            }

            let (agent_result, agent_messages) = match handle.await {
                Ok(Ok(msgs)) => {
                    let text = if output.is_empty() {
                        "(sub-agent produced no text output)".to_string()
                    } else {
                        output
                    };
                    (Ok(text), Some(msgs))
                }
                Ok(Err(e)) => (Err(format!("Sub-agent failed: {e}")), None),
                Err(e) => (Err(format!("Sub-agent panicked: {e}")), None),
            };

            // Handle worktree cleanup.
            let worktree_info = if let Some(wt) = &worktree {
                let has_changes = wt.has_changes().await;
                if has_changes {
                    let info = format!(
                        "\nworktree_path: {}\nworktree_branch: {}",
                        wt.path, wt.branch
                    );
                    Some(info)
                } else {
                    wt.cleanup().await;
                    None
                }
            } else {
                None
            };

            match agent_result {
                Ok(text) => {
                    // Store completed agent for potential resumption via SendMessage.
                    if let Some(msgs) = agent_messages {
                        let completed = super::CompletedAgent {
                            agent_id: agent_id.clone(),
                            name: agent_name.clone(),
                            messages: msgs,
                            model: model.clone(),
                            system: system.clone(),
                            tools: tools.clone(),
                            cwd: agent_cwd,
                        };
                        let mut agents = context.completed_agents.lock().await;
                        agents.insert(agent_id.clone(), completed);
                    }

                    let name_info = agent_name
                        .as_deref()
                        .map(|n| format!("\nagentName: {n}"))
                        .unwrap_or_default();
                    let wt_info = worktree_info.unwrap_or_default();
                    let result = format!(
                        "{text}\nagentId: {agent_id}{name_info}\n<usage>\ntotal_tokens: {total_tokens}\ntool_uses: {tool_use_count}\nduration_ms: {duration_ms}\n</usage>{wt_info}"
                    );
                    ToolResult::success(result)
                }
                Err(e) => {
                    if let Some(wt) = &worktree {
                        wt.cleanup().await;
                    }
                    ToolResult::error(e)
                }
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
            background_worktrees: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            completed_agents: Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
            parent_messages: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            cwd: None,
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
            .call(json!({"prompt": "test", "model": "haiku"}), &ctx)
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        // "haiku" should resolve to the full model ID.
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

    #[tokio::test]
    async fn invalid_isolation_rejected() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let (ctx, _bg_rx) = make_context(provider);

        let tool = AgentTool;
        let result = tool
            .call(json!({"prompt": "test", "isolation": "docker"}), &ctx)
            .await;
        assert!(result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("Unknown isolation mode"));
    }

    #[tokio::test]
    async fn worktree_lifecycle() {
        // Set up a temp git repo.
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_str().unwrap();
        let run = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .current_dir(dir_path)
                .output()
        };
        run(&["init"]).unwrap();
        run(&["config", "user.email", "test@test.com"]).unwrap();
        run(&["config", "user.name", "Test"]).unwrap();
        std::fs::write(dir.path().join("file.txt"), "hello").unwrap();
        run(&["add", "."]).unwrap();
        run(&["commit", "-m", "init"]).unwrap();

        // Create worktree, passing the repo path explicitly.
        let wt = Worktree::create(Some(dir_path))
            .await
            .expect("should create worktree");
        assert!(std::path::Path::new(&wt.path).exists());
        assert!(!wt.start_commit.is_empty());

        // No changes yet — has_changes should be false.
        assert!(!wt.has_changes().await);

        // Write a file in the worktree.
        std::fs::write(
            std::path::Path::new(&wt.path).join("new.txt"),
            "worktree change",
        )
        .unwrap();
        assert!(wt.has_changes().await);

        // Cleanup.
        let path = wt.path.clone();
        wt.cleanup().await;
        assert!(!std::path::Path::new(&path).exists());
    }

    #[tokio::test]
    async fn worktree_detects_committed_changes() {
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_str().unwrap();
        let run = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .current_dir(dir_path)
                .output()
        };
        run(&["init"]).unwrap();
        run(&["config", "user.email", "test@test.com"]).unwrap();
        run(&["config", "user.name", "Test"]).unwrap();
        std::fs::write(dir.path().join("file.txt"), "hello").unwrap();
        run(&["add", "."]).unwrap();
        run(&["commit", "-m", "init"]).unwrap();

        let wt = Worktree::create(Some(dir_path))
            .await
            .expect("should create worktree");

        // Commit a change inside the worktree.
        std::fs::write(std::path::Path::new(&wt.path).join("committed.txt"), "data").unwrap();
        let wt_path = wt.path.clone();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(&wt_path)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "wt commit"])
            .current_dir(&wt_path)
            .output()
            .unwrap();

        // git status is clean, but HEAD advanced — has_changes should be true.
        assert!(wt.has_changes().await);

        wt.cleanup().await;
    }

    #[tokio::test]
    async fn worktree_create_not_in_git_repo() {
        let dir = tempfile::tempdir().unwrap();
        // dir is a plain directory, not a git repo.
        let result = Worktree::create(Some(dir.path().to_str().unwrap())).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn worktree_cleanup_removes_branch() {
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_str().unwrap();
        let run = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .current_dir(dir_path)
                .output()
        };
        run(&["init"]).unwrap();
        run(&["config", "user.email", "test@test.com"]).unwrap();
        run(&["config", "user.name", "Test"]).unwrap();
        std::fs::write(dir.path().join("file.txt"), "hello").unwrap();
        run(&["add", "."]).unwrap();
        run(&["commit", "-m", "init"]).unwrap();

        let wt = Worktree::create(Some(dir_path))
            .await
            .expect("should create worktree");
        let branch = wt.branch.clone();

        wt.cleanup().await;

        // Verify branch no longer exists.
        let output = run(&["branch", "--list", &branch]).unwrap();
        let branches = String::from_utf8_lossy(&output.stdout);
        assert!(
            !branches.contains(&branch),
            "branch {branch} should be deleted after cleanup"
        );
    }

    #[tokio::test]
    async fn worktree_path_under_coo_dir() {
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_str().unwrap();
        let run = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .current_dir(dir_path)
                .output()
        };
        run(&["init"]).unwrap();
        run(&["config", "user.email", "test@test.com"]).unwrap();
        run(&["config", "user.name", "Test"]).unwrap();
        std::fs::write(dir.path().join("file.txt"), "hello").unwrap();
        run(&["add", "."]).unwrap();
        run(&["commit", "-m", "init"]).unwrap();

        let wt = Worktree::create(Some(dir_path))
            .await
            .expect("should create worktree");

        assert!(
            wt.path.contains(".coo/worktrees/"),
            "worktree path should be under .coo/worktrees/, got: {}",
            wt.path
        );

        wt.cleanup().await;
    }

    #[tokio::test]
    async fn pending_guard_decrements_on_drop() {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(1));
        {
            let _guard = crate::tools::PendingGuard(counter.clone());
            assert_eq!(counter.load(Ordering::SeqCst), 1);
        }
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn pending_guard_decrements_on_panic() {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(1));
        let c = counter.clone();
        let handle = tokio::spawn(async move {
            let _guard = crate::tools::PendingGuard(c);
            panic!("intentional panic");
        });
        let _ = handle.await; // JoinError::Panic
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "pending should be 0 after panic"
        );
    }

    #[tokio::test]
    async fn collect_agent_output_empty_channel() {
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(1);
        drop(tx); // Close immediately.
        let (output, tokens, tool_uses, had_error) = collect_agent_output(&mut rx).await;
        assert!(output.is_empty());
        assert_eq!(tokens, 0);
        assert_eq!(tool_uses, 0);
        assert!(!had_error);
    }

    #[tokio::test]
    async fn general_purpose_subagent_type_accepted() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let (mut ctx, _bg_rx) = make_context(provider);
        ctx.tools = Arc::new(ToolRegistry::with_defaults());

        let tool = AgentTool;
        let result = tool
            .call(
                json!({"prompt": "hello", "subagent_type": "general-purpose"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "general-purpose should be accepted");
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("hello"));
    }

    #[tokio::test]
    async fn unknown_subagent_type_rejected() {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let (ctx, _bg_rx) = make_context(provider);

        let tool = AgentTool;
        let result = tool
            .call(
                json!({"prompt": "test", "subagent_type": "nonexistent"}),
                &ctx,
            )
            .await;
        assert!(result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(text.contains("Unknown subagent_type"));
    }

    #[tokio::test]
    async fn explore_subagent_inherits_parent_model() {
        // Provider that echoes the model name.
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
        let (mut ctx, _bg_rx) = make_context(provider);
        ctx.tools = Arc::new(ToolRegistry::with_defaults());

        let tool = AgentTool;
        let result = tool
            .call(
                json!({"prompt": "search", "subagent_type": "explore"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        // explore inherits parent model ("test" from make_context)
        assert!(
            text.contains("test"),
            "explore should inherit parent model, got: {text}"
        );
    }

    #[tokio::test]
    async fn explore_subagent_denies_write_tools() {
        // Provider that tries to call the "write" tool.
        struct WriteAttemptProvider;

        #[async_trait]
        impl Provider for WriteAttemptProvider {
            async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
                let has_tool_result = req.messages.iter().any(|m| {
                    m.content
                        .iter()
                        .any(|c| matches!(c, ContentBlock::ToolResult { .. }))
                });
                if has_tool_result {
                    // Check if the tool result says "Unknown tool".
                    let error_text = req
                        .messages
                        .iter()
                        .flat_map(|m| &m.content)
                        .find_map(|c| {
                            if let ContentBlock::ToolResult { content, .. } = c {
                                content.first().map(|tc| match tc {
                                    crate::message::ToolResultContent::Text { text } => {
                                        text.clone()
                                    }
                                })
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    let _ = tx.send(Chunk::Text(error_text.clone())).await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::EndTurn,
                            content: vec![ContentBlock::Text { text: error_text }],
                        })
                        .await;
                } else {
                    let input = json!({"file_path": "/tmp/test", "content": "bad"});
                    let _ = tx
                        .send(Chunk::ToolUse {
                            id: "t1".into(),
                            name: "write".into(),
                            input: input.clone(),
                        })
                        .await;
                    let _ = tx
                        .send(Chunk::Done {
                            stop_reason: StopReason::ToolUse,
                            content: vec![ContentBlock::ToolUse {
                                id: "t1".into(),
                                name: "write".into(),
                                input,
                            }],
                        })
                        .await;
                }
                Ok(())
            }
        }

        let provider: Arc<dyn Provider> = Arc::new(WriteAttemptProvider);
        let (mut ctx, _bg_rx) = make_context(provider);
        ctx.tools = Arc::new(ToolRegistry::with_defaults());

        let tool = AgentTool;
        let result = tool
            .call(
                json!({"prompt": "write a file", "subagent_type": "explore"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(
            text.contains("Unknown tool"),
            "write should be denied for explore, got: {text}"
        );
    }

    #[tokio::test]
    async fn explicit_model_overrides_subagent_type() {
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
        let (mut ctx, _bg_rx) = make_context(provider);
        ctx.tools = Arc::new(ToolRegistry::with_defaults());

        let tool = AgentTool;
        let result = tool
            .call(
                json!({"prompt": "test", "subagent_type": "explore", "model": "my-custom-model"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let text = match &result.content[0] {
            ToolResultContent::Text { text } => text.as_str(),
        };
        assert!(
            text.contains("my-custom-model"),
            "explicit model should override subagent_type default, got: {text}"
        );
    }
}
