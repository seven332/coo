mod bash;
mod edit;
mod glob;
mod read;
mod write;

pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use read::ReadTool;
pub use write::WriteTool;

use async_trait::async_trait;

use crate::message::ToolResult;
use crate::provider::ToolDefinition;

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
    async fn call(&self, input: serde_json::Value) -> ToolResult;

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
        reg.register(Box::new(BashTool));
        reg.register(Box::new(EditTool));
        reg.register(Box::new(GlobTool));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_defaults() {
        let reg = ToolRegistry::with_defaults();
        assert!(reg.get("bash").is_some());
        assert!(reg.get("edit").is_some());
        assert!(reg.get("glob").is_some());
        assert!(reg.get("read").is_some());
        assert!(reg.get("write").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn definitions_match_tools() {
        let reg = ToolRegistry::with_defaults();
        let defs = reg.definitions();
        assert_eq!(defs.len(), 5);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"edit"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"read"));
        assert!(names.contains(&"write"));
    }
}
