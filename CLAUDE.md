# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Check

```bash
cargo build              # debug build
cargo build --release    # release build
cargo check              # type-check without building
cargo test               # run all tests
cargo test test_name     # run a single test
```

## What This Is

coo is a headless AI agent CLI written in Rust, designed to run in sandbox environments. It takes a prompt, runs an LLM-driven ReAct loop (think → tool_use → execute → feedback), and streams results as NDJSON to stdout. No terminal UI.

## Architecture

**Agent loop** (`src/agent.rs`): The core loop. Sends messages to the LLM, receives streaming chunks, executes tool calls, appends results back to the conversation, and repeats until the model stops requesting tools or hits max iterations. Uses `Arc<dyn Provider>` and `Arc<ToolRegistry>` to support sub-agent spawning.

**Provider abstraction** (`src/provider/`): The `Provider` trait defines `async fn request(&self, req, tx)` where `tx` is an mpsc channel for streaming `Chunk` events. Each LLM backend implements this trait. Providers: `AnthropicProvider` (SSE streaming), `MeowProvider` (random test provider).

**Tool system** (`src/tools/`): The `Tool` trait requires `name()`, `description()`, `input_schema()` (JSON Schema), and `async fn call(input, &ToolContext)`. `ToolContext` provides shared access to the provider, tool registry, model config, and depth tracking. Tools are registered in `ToolRegistry::with_defaults()`.

**Message types** (`src/message.rs`): `Message` (role + content blocks), `ContentBlock` (text/tool_use/tool_result/thinking), `ToolResult`, and `StreamEvent` (the NDJSON output events).

**Entry point** (`src/main.rs`): CLI arg parsing via clap, wires up provider + tools + agent, spawns the agent loop on tokio, and prints stream events as NDJSON lines.

## Commit Convention

Use [Conventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/). Format: `type(scope): description`.

## Key Design Decisions

- Provider streams chunks via `mpsc::Sender<Chunk>` rather than returning a collected response — enables real-time event forwarding.
- Tool results are sent back to the LLM as a `User` message containing `ToolResult` content blocks (following the Anthropic API conversation format).
- Provider and ToolRegistry are `Arc`-wrapped for sharing between parent agent and sub-agents spawned by the agent tool.
- Sub-agent depth is capped (default 5) to prevent infinite recursion.
- Bash tool uses `kill_on_drop(true)` to prevent orphan processes on timeout.
- Server-side tools (e.g. web_search) are passed through to the provider via `Request.server_tools`. The provider includes them in the API request; the server executes them and returns results inline. They are not executed locally.
- Logs go to stderr (`tracing`), NDJSON events go to stdout — clean separation for piping.
