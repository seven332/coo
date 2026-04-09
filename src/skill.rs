use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::{debug, warn};

/// A loaded skill definition.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
    pub allowed_tools: Option<String>,
    /// If set to "fork", the skill runs in an isolated sub-agent.
    pub context: Option<String>,
    /// Sub-agent type to use when context is "fork".
    pub agent: Option<String>,
    /// If true, only users can invoke this skill (not the model).
    pub disable_model_invocation: bool,
    /// If false, only the model can invoke this skill (not the user).
    pub user_invocable: bool,
}

/// Raw frontmatter parsed from YAML.
#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct Frontmatter {
    name: Option<String>,
    description: Option<String>,
    #[serde(rename = "allowed-tools")]
    allowed_tools: Option<String>,
    context: Option<String>,
    agent: Option<String>,
    #[serde(rename = "disable-model-invocation")]
    disable_model_invocation: bool,
    #[serde(rename = "user-invocable")]
    user_invocable: Option<bool>,
}

/// Registry of loaded skills, keyed by name.
#[derive(Debug, Default)]
pub struct SkillRegistry {
    skills: HashMap<String, Skill>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
        }
    }

    /// Load skills from the given working directory and user home.
    /// Project-level skills override user-level skills with the same name.
    pub fn load(cwd: Option<&str>) -> Self {
        let mut registry = Self::new();

        // User-level skills (lower priority — loaded first, overwritten by project).
        if let Some(home) = home_dir() {
            let user_dir = home.join(".coo");
            registry.load_from_dir(&user_dir);
        }

        // Project-level skills (higher priority — loaded second, overwrites user).
        if let Some(dir) = cwd {
            let project_dir = Path::new(dir).join(".coo");
            registry.load_from_dir(&project_dir);
        }

        debug!(count = registry.skills.len(), "Skills loaded");
        registry
    }

    /// Load skills from a `.coo` directory (both `skills/` and `commands/`).
    fn load_from_dir(&mut self, coo_dir: &Path) {
        // Load directory-based skills: .coo/skills/<name>/SKILL.md
        let skills_dir = coo_dir.join("skills");
        if skills_dir.is_dir()
            && let Ok(entries) = std::fs::read_dir(&skills_dir)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let skill_file = path.join("SKILL.md");
                    if skill_file.is_file() {
                        let fallback_name = entry.file_name().to_string_lossy().to_string();
                        if let Some(skill) = parse_skill_file(&skill_file, &fallback_name) {
                            self.skills.insert(skill.name.clone(), skill);
                        }
                    }
                }
            }
        }

        // Load single-file commands: .coo/commands/<name>.md
        let commands_dir = coo_dir.join("commands");
        if commands_dir.is_dir()
            && let Ok(entries) = std::fs::read_dir(&commands_dir)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().is_some_and(|ext| ext == "md") {
                    let fallback_name = path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    if let Some(skill) = parse_skill_file(&path, &fallback_name) {
                        self.skills.insert(skill.name.clone(), skill);
                    }
                }
            }
        }
    }

    /// Get a skill by name.
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    /// List all loaded skills.
    pub fn list(&self) -> Vec<&Skill> {
        self.skills.values().collect()
    }

    /// Check if any skills are loaded.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

/// Parse a skill file (SKILL.md or <name>.md) into a Skill.
fn parse_skill_file(path: &Path, fallback_name: &str) -> Option<Skill> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "Failed to read skill file");
            return None;
        }
    };

    let (frontmatter, content) = parse_frontmatter(&raw);

    let fm: Frontmatter = match &frontmatter {
        Some(yaml) => match serde_yaml::from_str(yaml) {
            Ok(fm) => fm,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "Failed to parse skill frontmatter");
                Frontmatter::default()
            }
        },
        None => Frontmatter::default(),
    };

    let name = fm.name.unwrap_or_else(|| fallback_name.to_string());
    let description = fm.description.unwrap_or_default();

    Some(Skill {
        name,
        description,
        content: content.to_string(),
        allowed_tools: fm.allowed_tools,
        context: fm.context,
        agent: fm.agent,
        disable_model_invocation: fm.disable_model_invocation,
        user_invocable: fm.user_invocable.unwrap_or(true),
    })
}

/// Split a markdown file into optional YAML frontmatter and body content.
/// Frontmatter is delimited by `---` on its own line at the start of the file.
fn parse_frontmatter(raw: &str) -> (Option<String>, &str) {
    let trimmed = raw.trim_start();
    if !trimmed.starts_with("---") {
        return (None, raw);
    }

    // Find the closing `---` after the opening one.
    let after_open = &trimmed[3..];
    let after_open = after_open.strip_prefix('\n').unwrap_or(after_open);

    // Find closing `---`: either at the very start (empty frontmatter) or after a newline.
    let close_pos = if after_open.starts_with("---") {
        Some(0)
    } else {
        after_open.find("\n---").map(|p| p + 1) // +1 to skip the newline
    };

    if let Some(pos) = close_pos {
        let yaml = &after_open[..pos];
        let rest = &after_open[pos + 3..]; // skip "---"
        let rest = rest.strip_prefix('\n').unwrap_or(rest);
        (Some(yaml.to_string()), rest)
    } else {
        // No closing ---, treat as no frontmatter.
        (None, raw)
    }
}

/// Get the user's home directory.
fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn parse_frontmatter_with_yaml() {
        let raw = "---\nname: test\ndescription: A test skill\n---\nHello world";
        let (fm, content) = parse_frontmatter(raw);
        let fm = fm.unwrap();
        assert!(fm.contains("name: test"));
        assert!(fm.contains("description: A test skill"));
        assert_eq!(content, "Hello world");
    }

    #[test]
    fn parse_frontmatter_no_yaml() {
        let raw = "Just some content";
        let (fm, content) = parse_frontmatter(raw);
        assert!(fm.is_none());
        assert_eq!(content, "Just some content");
    }

    #[test]
    fn parse_frontmatter_no_closing() {
        let raw = "---\nname: test\nno closing";
        let (fm, content) = parse_frontmatter(raw);
        assert!(fm.is_none());
        assert_eq!(content, raw);
    }

    #[test]
    fn parse_frontmatter_empty_yaml() {
        let raw = "---\n---\nContent here";
        let (fm, content) = parse_frontmatter(raw);
        assert_eq!(fm.unwrap(), "");
        assert_eq!(content, "Content here");
    }

    #[test]
    fn parse_skill_file_full() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.md");
        fs::write(
            &path,
            "---\nname: deploy\ndescription: Deploy the app\nallowed-tools: Bash(git *)\ncontext: fork\nagent: Explore\ndisable-model-invocation: true\nuser-invocable: false\n---\nDeploy $ARGUMENTS to production.",
        )
        .unwrap();

        let skill = parse_skill_file(&path, "fallback").unwrap();
        assert_eq!(skill.name, "deploy");
        assert_eq!(skill.description, "Deploy the app");
        assert_eq!(skill.content, "Deploy $ARGUMENTS to production.");
        assert_eq!(skill.allowed_tools.as_deref(), Some("Bash(git *)"));
        assert_eq!(skill.context.as_deref(), Some("fork"));
        assert_eq!(skill.agent.as_deref(), Some("Explore"));
        assert!(skill.disable_model_invocation);
        assert!(!skill.user_invocable);
    }

    #[test]
    fn parse_skill_file_minimal() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("simple.md");
        fs::write(&path, "Just a simple skill with no frontmatter.").unwrap();

        let skill = parse_skill_file(&path, "simple").unwrap();
        assert_eq!(skill.name, "simple");
        assert_eq!(skill.description, "");
        assert_eq!(skill.content, "Just a simple skill with no frontmatter.");
        assert!(skill.allowed_tools.is_none());
        assert!(skill.context.is_none());
        assert!(!skill.disable_model_invocation);
        assert!(skill.user_invocable);
    }

    #[test]
    fn parse_skill_file_name_from_frontmatter_overrides_fallback() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("file.md");
        fs::write(&path, "---\nname: custom-name\n---\nContent").unwrap();

        let skill = parse_skill_file(&path, "fallback").unwrap();
        assert_eq!(skill.name, "custom-name");
    }

    #[test]
    fn parse_skill_file_fallback_name_when_no_frontmatter_name() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("file.md");
        fs::write(&path, "---\ndescription: test\n---\nContent").unwrap();

        let skill = parse_skill_file(&path, "my-fallback").unwrap();
        assert_eq!(skill.name, "my-fallback");
    }

    #[test]
    fn load_skills_from_directory() {
        let dir = TempDir::new().unwrap();
        let coo_dir = dir.path().join(".coo");

        // Create a directory-based skill.
        let skill_dir = coo_dir.join("skills").join("deploy");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: deploy\ndescription: Deploy app\n---\nDeploy now.",
        )
        .unwrap();

        // Create a command-based skill.
        let cmd_dir = coo_dir.join("commands");
        fs::create_dir_all(&cmd_dir).unwrap();
        fs::write(
            cmd_dir.join("lint.md"),
            "---\ndescription: Run linter\n---\nRun lint.",
        )
        .unwrap();

        let registry = SkillRegistry::load(Some(dir.path().to_str().unwrap()));
        assert_eq!(registry.list().len(), 2);
        assert!(registry.get("deploy").is_some());
        assert!(registry.get("lint").is_some());

        let deploy = registry.get("deploy").unwrap();
        assert_eq!(deploy.description, "Deploy app");
        assert_eq!(deploy.content, "Deploy now.");

        let lint = registry.get("lint").unwrap();
        assert_eq!(lint.description, "Run linter");
        assert_eq!(lint.content, "Run lint.");
    }

    #[test]
    fn project_overrides_user_skills() {
        let project_dir = TempDir::new().unwrap();
        let user_dir = TempDir::new().unwrap();

        // User-level skill.
        let user_coo = user_dir.path().join(".coo").join("commands");
        fs::create_dir_all(&user_coo).unwrap();
        fs::write(
            user_coo.join("greet.md"),
            "---\ndescription: User greeting\n---\nHello from user.",
        )
        .unwrap();

        // Project-level skill with same name.
        let project_coo = project_dir.path().join(".coo").join("commands");
        fs::create_dir_all(&project_coo).unwrap();
        fs::write(
            project_coo.join("greet.md"),
            "---\ndescription: Project greeting\n---\nHello from project.",
        )
        .unwrap();

        // Simulate loading: user first, then project overwrites.
        let mut registry = SkillRegistry::new();
        registry.load_from_dir(&user_dir.path().join(".coo"));
        registry.load_from_dir(&project_dir.path().join(".coo"));

        let greet = registry.get("greet").unwrap();
        assert_eq!(greet.description, "Project greeting");
        assert_eq!(greet.content, "Hello from project.");
    }

    #[test]
    fn empty_directory_loads_no_skills() {
        let dir = TempDir::new().unwrap();
        let registry = SkillRegistry::load(Some(dir.path().to_str().unwrap()));
        assert!(registry.is_empty());
    }

    #[test]
    fn non_md_files_ignored() {
        let dir = TempDir::new().unwrap();
        let cmd_dir = dir.path().join(".coo").join("commands");
        fs::create_dir_all(&cmd_dir).unwrap();
        fs::write(cmd_dir.join("readme.txt"), "Not a skill").unwrap();
        fs::write(cmd_dir.join("script.sh"), "#!/bin/bash").unwrap();

        let registry = SkillRegistry::load(Some(dir.path().to_str().unwrap()));
        assert!(registry.is_empty());
    }

    #[test]
    fn skill_dir_without_skill_md_ignored() {
        let dir = TempDir::new().unwrap();
        let skill_dir = dir.path().join(".coo").join("skills").join("broken");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(skill_dir.join("README.md"), "Not SKILL.md").unwrap();

        let registry = SkillRegistry::load(Some(dir.path().to_str().unwrap()));
        assert!(registry.is_empty());
    }
}
