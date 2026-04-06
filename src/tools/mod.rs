mod agent_tool;
mod bash;
mod edit;
mod glob;
mod grep;
mod read;
mod web_fetch;
mod write;

pub use agent_tool::AgentTool;
pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read::ReadTool;
pub use web_fetch::WebFetchTool;
pub use write::WriteTool;

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use async_trait::async_trait;
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

use crate::message::ToolResult;
use crate::provider::{Provider, ToolDefinition};

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
        reg.register(Box::new(WebFetchTool::new()));
        reg.register(Box::new(WriteTool));
        reg
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.to_definition()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| &**t)
    }
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
        assert!(reg.get("web_fetch").is_some());
        assert!(reg.get("write").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn definitions_match_tools() {
        let reg = ToolRegistry::with_defaults();
        let defs = reg.definitions();
        assert_eq!(defs.len(), 8);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"agent"));
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"edit"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"read"));
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
}
