mod agent;
mod message;
mod provider;
mod tools;

use clap::Parser;
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;

use agent::Agent;
use message::StreamEvent;
use provider::AnthropicProvider;
use tools::ToolRegistry;

#[derive(Parser)]
#[command(name = "you-mind", about = "An agent CLI for sandbox environments")]
struct Cli {
    /// The prompt to send to the agent.
    #[arg(short, long)]
    prompt: Option<String>,

    /// Model to use.
    #[arg(short, long, default_value = "claude-sonnet-4-20250514")]
    model: String,

    /// System prompt.
    #[arg(short, long)]
    system: Option<String>,

    /// API base URL.
    #[arg(long, env = "ANTHROPIC_BASE_URL")]
    base_url: Option<String>,

    /// API key.
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    api_key: Option<String>,

    /// Max output tokens per LLM call.
    #[arg(long, default_value = "16384")]
    max_tokens: u32,

    /// Read prompt from stdin.
    #[arg(long)]
    stdin: bool,
}

const DEFAULT_SYSTEM: &str = "\
You are an AI assistant running in a sandbox environment. \
You have access to tools for executing bash commands, reading files, and writing files. \
Use these tools to accomplish the user's tasks. \
Be concise and direct.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let api_key = cli.api_key.unwrap_or_else(|| {
        std::env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| {
            eprintln!("Error: ANTHROPIC_API_KEY is required");
            std::process::exit(1);
        })
    });

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

    let provider = AnthropicProvider::new(api_key, cli.base_url);
    let tools = ToolRegistry::with_defaults();
    let system = cli.system.unwrap_or_else(|| DEFAULT_SYSTEM.to_string());

    let mut agent = Agent::new(Box::new(provider), tools, cli.model, system);
    agent.max_tokens = cli.max_tokens;

    let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(128);

    // Spawn the agent loop.
    let agent_handle = tokio::spawn(async move {
        agent.run(prompt, event_tx).await
    });

    // Stream events as NDJSON to stdout.
    while let Some(event) = event_rx.recv().await {
        let json = serde_json::to_string(&event)?;
        println!("{json}");
    }

    agent_handle.await??;
    Ok(())
}
