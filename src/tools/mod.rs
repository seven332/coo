mod agent_tool;
mod bash;
mod edit;
mod glob;
mod grep;
mod read;
mod send_message;
mod web_fetch;
mod write;

pub use agent_tool::AgentTool;
pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read::ReadTool;
pub use send_message::SendMessageTool;
pub use web_fetch::WebFetchTool;
pub use write::WriteTool;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use async_trait::async_trait;
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

use crate::message::{StreamEvent, ToolResult};
use crate::provider::{Provider, ToolDefinition};

/// Collect text output, token usage, tool use count, and error status from agent events.
/// Returns (output, total_tokens, tool_use_count, had_error).
/// Shared by AgentTool and SendMessageTool.
pub(crate) async fn collect_agent_output(
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

/// Create a git Command with inherited git env vars removed.
/// Prevents lefthook/pre-commit GIT_DIR, GIT_INDEX_FILE, etc.
/// from leaking into sub-agent git operations.
pub(crate) fn git_cmd() -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new("git");
    for var in [
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ] {
        cmd.env_remove(var);
    }
    cmd
}

/// Result from a background agent that has completed.
pub struct BackgroundAgentResult {
    pub agent_id: String,
    pub description: String,
    pub result: String,
    pub is_error: bool,
    pub duration_ms: u64,
    pub total_tokens: u64,
    pub tool_use_count: usize,
}

/// State of a completed sub-agent, stored for resumption via SendMessage.
pub struct CompletedAgent {
    pub agent_id: String,
    pub name: Option<String>,
    pub messages: Vec<crate::message::Message>,
    pub model: String,
    pub system: String,
    pub tools: Arc<ToolRegistry>,
    pub cwd: Option<String>,
}

/// Info needed to clean up an orphaned worktree.
pub struct WorktreeInfo {
    pub path: String,
    pub branch: String,
    pub git_root: String,
}

/// Context passed to tools during execution.
pub struct ToolContext {
    pub provider: Arc<dyn Provider>,
    pub tools: Arc<ToolRegistry>,
    pub model: String,
    pub system: String,
    pub max_tokens: u32,
    pub max_iterations: usize,
    pub depth: usize,
    pub max_depth: usize,
    pub background_tx: mpsc::Sender<BackgroundAgentResult>,
    pub pending_background: Arc<AtomicUsize>,
    pub background_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    /// Worktrees created by background agents, for cleanup on abort.
    pub background_worktrees: Arc<Mutex<Vec<WorktreeInfo>>>,
    /// Completed sub-agents available for resumption via SendMessage.
    /// Keyed by agent_id. Agents with names are also findable by name.
    pub completed_agents: Arc<Mutex<HashMap<String, CompletedAgent>>>,
    /// Parent agent's messages for fork optimization (prompt cache reuse).
    /// Updated each iteration with the current conversation state.
    pub parent_messages: Arc<Mutex<Vec<crate::message::Message>>>,
    /// Working directory for tool execution. If None, uses process CWD.
    pub cwd: Option<String>,
}

impl ToolContext {
    /// Resolve a file path relative to the tool context's working directory.
    /// Absolute paths are returned as-is; relative paths are resolved against `cwd`.
    pub fn resolve_path(&self, path: &str) -> std::path::PathBuf {
        let p = std::path::Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else if let Some(ref cwd) = self.cwd {
            std::path::Path::new(cwd).join(p)
        } else {
            p.to_path_buf()
        }
    }
}

/// Trait that all tools must implement.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool name (used in API tool_use blocks).
    fn name(&self) -> &str;

    /// Description for the LLM.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input parameters.
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given input.
    async fn call(&self, input: serde_json::Value, context: &ToolContext) -> ToolResult;

    /// Convert to API tool definition.
    fn to_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Create a registry with the default set of tools.
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(AgentTool));
        reg.register(Box::new(BashTool));
        reg.register(Box::new(EditTool));
        reg.register(Box::new(GlobTool));
        reg.register(Box::new(GrepTool));
        reg.register(Box::new(ReadTool));
        reg.register(Box::new(SendMessageTool));
        reg.register(Box::new(WebFetchTool::new()));
        reg.register(Box::new(WriteTool));
        reg
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Create a new registry containing only tools whose names pass the filter.
    pub fn filtered(&self, allow: impl Fn(&str) -> bool) -> Self {
        let mut reg = Self::new();
        for tool_name in self.tools.iter().map(|t| t.name()) {
            if allow(tool_name) {
                // Re-create default tools by name. This works because all tools
                // are stateless singletons (except WebFetchTool which is cheap).
                if let Some(tool) = create_tool_by_name(tool_name) {
                    reg.register(tool);
                }
            }
        }
        reg
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.to_definition()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| &**t)
    }
}

/// Create a tool instance by name. Returns None for unknown names.
fn create_tool_by_name(name: &str) -> Option<Box<dyn Tool>> {
    match name {
        "agent" => Some(Box::new(AgentTool)),
        "bash" => Some(Box::new(BashTool)),
        "edit" => Some(Box::new(EditTool)),
        "glob" => Some(Box::new(GlobTool)),
        "grep" => Some(Box::new(GrepTool)),
        "read" => Some(Box::new(ReadTool)),
        "send_message" => Some(Box::new(SendMessageTool)),
        "web_fetch" => Some(Box::new(WebFetchTool::new())),
        "write" => Some(Box::new(WriteTool)),
        _ => None,
    }
}

/// Built-in sub-agent type definition.
pub struct SubAgentType {
    pub name: &'static str,
    pub description: &'static str,
    /// Model to use. None means inherit from parent.
    pub model: Option<&'static str>,
    /// System prompt override. None means inherit from parent.
    pub system: Option<&'static str>,
    /// Tool names to deny. All other tools are allowed.
    pub denied_tools: &'static [&'static str],
}

/// Built-in sub-agent types.
pub static SUBAGENT_TYPES: &[SubAgentType] = &[
    SubAgentType {
        name: "explore",
        description: "Fast agent for codebase exploration and search. Cannot modify files.",
        model: None,
        system: Some(
            "You are a fast codebase exploration agent. Your job is to find files, search code, \
             and answer questions about the codebase. You cannot modify files. Be concise and direct.",
        ),
        denied_tools: &["agent", "edit", "write"],
    },
    SubAgentType {
        name: "plan",
        description: "Software architect agent for designing implementation plans. Cannot modify files.",
        model: None,
        system: Some(
            "You are a software architect agent. Your job is to research the codebase and design \
             implementation plans. You cannot modify files. Return step-by-step plans with file \
             paths and key considerations.",
        ),
        denied_tools: &["agent", "edit", "write"],
    },
];

/// Look up a built-in sub-agent type by name.
pub fn get_subagent_type(name: &str) -> Option<&'static SubAgentType> {
    SUBAGENT_TYPES.iter().find(|t| t.name == name)
}

/// Create a dummy ToolContext for tests where context is not needed.
#[cfg(test)]
pub(crate) fn dummy_context() -> ToolContext {
    use crate::provider::{Chunk, Provider, Request};

    struct NoopProvider;

    #[async_trait]
    impl Provider for NoopProvider {
        async fn request(
            &self,
            _req: Request,
            _tx: tokio::sync::mpsc::Sender<Chunk>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    let (background_tx, _) = mpsc::channel(16);
    ToolContext {
        provider: Arc::new(NoopProvider),
        tools: Arc::new(ToolRegistry::new()),
        model: "test".into(),
        system: "test".into(),
        max_tokens: 1024,
        max_iterations: 100,
        depth: 0,
        max_depth: 5,
        background_tx,
        pending_background: Arc::new(AtomicUsize::new(0)),
        background_handles: Arc::new(Mutex::new(Vec::new())),
        background_worktrees: Arc::new(Mutex::new(Vec::new())),
        completed_agents: Arc::new(Mutex::new(HashMap::new())),
        parent_messages: Arc::new(Mutex::new(Vec::new())),
        cwd: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_defaults() {
        let reg = ToolRegistry::with_defaults();
        assert!(reg.get("agent").is_some());
        assert!(reg.get("bash").is_some());
        assert!(reg.get("edit").is_some());
        assert!(reg.get("glob").is_some());
        assert!(reg.get("grep").is_some());
        assert!(reg.get("read").is_some());
        assert!(reg.get("send_message").is_some());
        assert!(reg.get("web_fetch").is_some());
        assert!(reg.get("write").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn definitions_match_tools() {
        let reg = ToolRegistry::with_defaults();
        let defs = reg.definitions();
        assert_eq!(defs.len(), 9);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"agent"));
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"edit"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"read"));
        assert!(names.contains(&"send_message"));
        assert!(names.contains(&"web_fetch"));
        assert!(names.contains(&"write"));
    }

    #[test]
    fn resolve_path_absolute() {
        let ctx = dummy_context();
        assert_eq!(
            ctx.resolve_path("/tmp/foo.txt"),
            std::path::PathBuf::from("/tmp/foo.txt")
        );
    }

    #[test]
    fn resolve_path_relative_no_cwd() {
        let ctx = dummy_context();
        assert_eq!(
            ctx.resolve_path("foo.txt"),
            std::path::PathBuf::from("foo.txt")
        );
    }

    #[test]
    fn resolve_path_relative_with_cwd() {
        let mut ctx = dummy_context();
        ctx.cwd = Some("/work/dir".into());
        assert_eq!(
            ctx.resolve_path("foo.txt"),
            std::path::PathBuf::from("/work/dir/foo.txt")
        );
    }

    #[test]
    fn resolve_path_absolute_ignores_cwd() {
        let mut ctx = dummy_context();
        ctx.cwd = Some("/work/dir".into());
        assert_eq!(
            ctx.resolve_path("/tmp/foo.txt"),
            std::path::PathBuf::from("/tmp/foo.txt")
        );
    }

    #[test]
    fn filtered_registry() {
        let reg = ToolRegistry::with_defaults();
        let filtered = reg.filtered(|name| name != "write" && name != "edit");
        assert!(filtered.get("read").is_some());
        assert!(filtered.get("glob").is_some());
        assert!(filtered.get("write").is_none());
        assert!(filtered.get("edit").is_none());
    }

    #[test]
    fn create_tool_by_name_covers_all_defaults() {
        let reg = ToolRegistry::with_defaults();
        for def in reg.definitions() {
            assert!(
                create_tool_by_name(&def.name).is_some(),
                "create_tool_by_name missing tool: {}",
                def.name
            );
        }
    }

    #[test]
    fn get_subagent_type_found() {
        assert!(get_subagent_type("explore").is_some());
        assert!(get_subagent_type("plan").is_some());
    }

    #[test]
    fn get_subagent_type_not_found() {
        assert!(get_subagent_type("nonexistent").is_none());
    }

    #[test]
    fn explore_type_config() {
        let t = get_subagent_type("explore").unwrap();
        assert!(t.denied_tools.contains(&"write"));
        assert!(t.denied_tools.contains(&"edit"));
        assert!(t.denied_tools.contains(&"agent"));
        assert!(!t.denied_tools.contains(&"bash")); // bash allowed
        assert!(t.model.is_none()); // inherits parent model
    }

    #[test]
    fn plan_type_config() {
        let t = get_subagent_type("plan").unwrap();
        assert!(t.model.is_none());
        assert!(t.denied_tools.contains(&"write"));
        assert!(t.denied_tools.contains(&"edit"));
        assert!(!t.denied_tools.contains(&"bash")); // bash allowed
    }
}
