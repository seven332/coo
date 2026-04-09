use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::message::ToolResult;

use super::{Tool, ToolContext};

pub struct SkillTool;

#[derive(Deserialize)]
struct Input {
    name: String,
    #[serde(default)]
    args: String,
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &str {
        "skill"
    }

    fn description(&self) -> &str {
        "Invoke a loaded skill by name. Skills are prompt-based extensions loaded from \
         .coo/skills/ and .coo/commands/. The skill content is rendered with argument \
         substitution and returned."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name to invoke"
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments for the skill",
                    "default": ""
                }
            },
            "required": ["name"]
        })
    }

    async fn call(&self, input: serde_json::Value, context: &ToolContext) -> ToolResult {
        let input: Input = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };

        let skill = match context.skills.get(&input.name) {
            Some(s) => s,
            None => {
                let available: Vec<&str> = context
                    .skills
                    .list()
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect();
                if available.is_empty() {
                    return ToolResult::error(format!(
                        "Skill '{}' not found. No skills are loaded.",
                        input.name
                    ));
                }
                return ToolResult::error(format!(
                    "Skill '{}' not found. Available skills: {}",
                    input.name,
                    available.join(", ")
                ));
            }
        };

        if skill.disable_model_invocation {
            return ToolResult::error(format!(
                "Skill '{}' has disabled model invocation.",
                input.name
            ));
        }

        let rendered = skill.render(&input.args);
        ToolResult::success(rendered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill::{Skill, SkillRegistry};
    use std::sync::Arc;

    fn make_context_with_skills(skills: Vec<Skill>) -> ToolContext {
        let mut registry = SkillRegistry::new();
        for skill in skills {
            registry.insert(skill);
        }
        let mut ctx = crate::tools::dummy_context();
        ctx.skills = Arc::new(registry);
        ctx
    }

    fn make_skill(name: &str, content: &str) -> Skill {
        Skill {
            name: name.into(),
            description: format!("Test skill: {name}"),
            content: content.into(),
            allowed_tools: None,
            context: None,
            agent: None,
            disable_model_invocation: false,
            user_invocable: true,
        }
    }

    #[tokio::test]
    async fn invoke_skill_with_args() {
        let ctx = make_context_with_skills(vec![make_skill("greet", "Hello $0!")]);
        let result = SkillTool
            .call(json!({"name": "greet", "args": "world"}), &ctx)
            .await;
        assert!(!result.is_error);
        assert_eq!(result.content[0].text(), Some("Hello world!"));
    }

    #[tokio::test]
    async fn invoke_skill_without_args() {
        let ctx = make_context_with_skills(vec![make_skill("status", "Show status.")]);
        let result = SkillTool.call(json!({"name": "status"}), &ctx).await;
        assert!(!result.is_error);
        assert_eq!(result.content[0].text(), Some("Show status."));
    }

    #[tokio::test]
    async fn skill_not_found() {
        let ctx = make_context_with_skills(vec![make_skill("deploy", "Deploy.")]);
        let result = SkillTool.call(json!({"name": "nonexistent"}), &ctx).await;
        assert!(result.is_error);
        let text = result.content[0].text().unwrap();
        assert!(text.contains("not found"));
        assert!(text.contains("deploy"));
    }

    #[tokio::test]
    async fn skill_not_found_no_skills() {
        let ctx = make_context_with_skills(vec![]);
        let result = SkillTool.call(json!({"name": "anything"}), &ctx).await;
        assert!(result.is_error);
        let text = result.content[0].text().unwrap();
        assert!(text.contains("No skills are loaded"));
    }

    #[tokio::test]
    async fn model_invocation_disabled() {
        let skill = Skill {
            disable_model_invocation: true,
            ..make_skill("secret", "Secret stuff.")
        };
        let ctx = make_context_with_skills(vec![skill]);
        let result = SkillTool.call(json!({"name": "secret"}), &ctx).await;
        assert!(result.is_error);
        let text = result.content[0].text().unwrap();
        assert!(text.contains("disabled model invocation"));
    }

    #[tokio::test]
    async fn invalid_input() {
        let ctx = make_context_with_skills(vec![]);
        let result = SkillTool.call(json!({}), &ctx).await;
        assert!(result.is_error);
    }
}
