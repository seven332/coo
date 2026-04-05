use std::sync::Arc;

use clap::Parser;
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;

use coo::agent::Agent;
use coo::message::StreamEvent;
use coo::provider::{AnthropicProvider, MeowProvider, Provider, ServerTool};
use coo::tools::ToolRegistry;

#[derive(Parser)]
#[command(name = "coo", about = "An agent CLI for sandbox environments")]
struct Cli {
    /// The prompt to send to the agent.
    #[arg(short, long)]
    prompt: Option<String>,

    /// Model to use.
    #[arg(short, long, default_value = "claude-sonnet-4-20250514")]
    model: String,

    /// Provider to use: "anthropic" or "meow".
    #[arg(long, default_value = "anthropic")]
    provider: String,

    /// System prompt.
    #[arg(short, long)]
    system: Option<String>,

    /// API base URL.
    #[arg(long, env = "ANTHROPIC_BASE_URL")]
    base_url: Option<String>,

    /// API key (required for anthropic provider, or set ANTHROPIC_API_KEY).
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    api_key: Option<String>,

    /// Max output tokens per LLM call.
    #[arg(long, default_value = "32000")]
    max_tokens: u32,

    /// Enable web search (Anthropic provider only).
    #[arg(long)]
    web_search: bool,

    /// Read prompt from stdin.
    #[arg(long)]
    stdin: bool,
}

const DEFAULT_SYSTEM: &str = r#"You are an AI assistant running in a sandbox environment. A user is communicating with you through a web or chat interface. You can act freely in the sandbox — no permissions or confirmations needed.

# Environment
- You are in a disposable sandbox. Full, unrestricted access. Safe to experiment.
- Do not ask for confirmation before acting. The user expects you to just do it.

# Tools
Use the right tool for the job:
- `bash`: run commands, install packages, manage processes
- `read`: read files (not cat)
- `write`: create files (not echo/cat)
- `edit`: modify files with exact string replacement (not sed)
- `glob`: find files by pattern (not find/ls)
- `grep`: search file contents (not grep/rg)
- `web_fetch`: fetch URLs, HTML is converted to markdown
- `agent`: spawn a sub-agent for independent sub-tasks

Call multiple tools in parallel when they are independent of each other.

# Approach
- Act first, then report results. Do not announce what you are about to do.
- If something fails, diagnose the error and try a different approach. Do not give up after one failure.
- For complex tasks, break them into steps. Use the agent tool to parallelize independent work.
- Verify your work — run tests, check outputs, confirm results.
- Be concise. Lead with results, not reasoning. The user can see your tool calls."#;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let prompt = if cli.stdin {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else if let Some(p) = cli.prompt {
        p
    } else {
        eprintln!("Error: provide --prompt or --stdin");
        std::process::exit(1);
    };

    let provider: Arc<dyn Provider> = match cli.provider.as_str() {
        "meow" => Arc::new(MeowProvider),
        "anthropic" => {
            let api_key = cli.api_key.unwrap_or_else(|| {
                eprintln!("Error: ANTHROPIC_API_KEY is required for anthropic provider");
                std::process::exit(2);
            });
            Arc::new(AnthropicProvider::new(api_key, cli.base_url))
        }
        other => {
            eprintln!("Error: unknown provider: {other}");
            std::process::exit(1);
        }
    };

    let tools = Arc::new(ToolRegistry::with_defaults());
    let system = cli.system.unwrap_or_else(|| DEFAULT_SYSTEM.to_string());

    let mut agent = Agent::new(provider, tools, cli.model, system);
    agent.max_tokens = cli.max_tokens;
    if cli.web_search {
        agent.server_tools.push(ServerTool::WebSearch {
            name: "web_search".into(),
            max_uses: Some(8),
        });
    }

    let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(128);

    // Spawn the agent loop.
    let agent_handle = tokio::spawn(async move { agent.run(prompt, event_tx).await });

    // Stream events as NDJSON to stdout.
    while let Some(event) = event_rx.recv().await {
        let json = serde_json::to_string(&event)?;
        println!("{json}");
    }

    agent_handle.await??;
    Ok(())
}
