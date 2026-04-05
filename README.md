# coo

CooCoo is an AI agent CLI designed for sandbox environments. Similar to Claude Code, but runs headless with NDJSON streaming output — no terminal UI needed.

## Usage

```bash
# Direct prompt
coo --prompt "list all files in /tmp"

# Read prompt from stdin
echo "create a hello.py file" | coo --stdin

# Custom model and system prompt
coo --prompt "fix the bug" --model claude-sonnet-4-6 --system "You are a Rust expert."

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

Output follows the Claude Code `stream-json` format (NDJSON):

```jsonl
{"type":"system","subtype":"init","tools":["bash","read",...],"model":"claude-sonnet-4-6","cwd":"/workspace","session_id":"...","uuid":"..."}
{"type":"user","message":{"role":"user","content":[{"type":"text","text":"list files"}]},"parent_tool_use_id":null,"session_id":"...","uuid":"..."}
{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"Here are"}},"parent_tool_use_id":null,"session_id":"...","uuid":"..."}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Here are the files."}]},"parent_tool_use_id":null,"session_id":"...","uuid":"..."}
{"type":"result","subtype":"success","duration_ms":1234,"is_error":false,"num_turns":1,"result":"Here are the files.","session_id":"...","uuid":"..."}
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
