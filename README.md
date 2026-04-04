# coo

An AI agent CLI designed for sandbox environments. Similar to Claude Code, but runs headless with NDJSON streaming output — no terminal UI needed.

## Usage

```bash
# Direct prompt
coo --prompt "list all files in /tmp"

# Read prompt from stdin
echo "create a hello.py file" | coo --stdin

# Custom model and system prompt
coo --prompt "fix the bug" --model claude-sonnet-4-20250514 --system "You are a Rust expert."

# Enable web search (Anthropic provider only)
coo --web-search --prompt "what is the latest Rust version?"

# Use the meow provider for testing (no API key needed)
coo --provider meow --prompt "hello"
```

Output is streamed as NDJSON (one JSON event per line) to stdout.

## Environment Variables

- `ANTHROPIC_API_KEY` — API key (required for anthropic provider)
- `ANTHROPIC_BASE_URL` — Custom API endpoint

## Event Types

```jsonl
{"type":"thinking","thinking":"Let me analyze..."}
{"type":"text","text":"I'll create the file."}
{"type":"tool_use","id":"toolu_01","name":"bash","input":{"command":"ls"}}
{"type":"tool_result","tool_use_id":"toolu_01","content":"file1\nfile2","is_error":false}
{"type":"done","reason":"end_turn"}
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `agent` | Spawn sub-agents for complex tasks |
| `bash` | Execute shell commands |
| `edit` | Exact string replacement in files |
| `glob` | Find files by pattern |
| `grep` | Search file contents (uses rg/grep) |
| `read` | Read files with line numbers |
| `web_fetch` | Fetch URLs and convert to Markdown |
| `web_search` | Search the web (Anthropic server-side, `--web-search`) |
| `write` | Write/create files |

## Build

```bash
cargo build --release
```
