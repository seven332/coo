mod agent_tool;
mod bash;
mod edit;
mod glob;
mod grep;
mod read;
mod write;

pub use agent_tool::AgentTool;
pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read::ReadTool;
pub use write::WriteTool;

use std::sync::Arc;

use async_trait::async_trait;

use crate::message::ToolResult;
use crate::provider::{Provider, ToolDefinition};

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

    ToolContext {
        provider: Arc::new(NoopProvider),
        tools: Arc::new(ToolRegistry::new()),
        model: "test".into(),
        system: "test".into(),
        max_tokens: 1024,
        max_iterations: 100,
        depth: 0,
        max_depth: 5,
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
        assert!(reg.get("write").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn definitions_match_tools() {
        let reg = ToolRegistry::with_defaults();
        let defs = reg.definitions();
        assert_eq!(defs.len(), 7);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"agent"));
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"edit"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"read"));
        assert!(names.contains(&"write"));
    }
}
