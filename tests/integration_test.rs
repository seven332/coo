use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::mpsc;

use you_mind::agent::Agent;
use you_mind::message::{ContentBlock, StreamEvent};
use you_mind::provider::{Chunk, Provider, Request, StopReason};
use you_mind::tools::ToolRegistry;

// --- Fake Provider ---

/// A scriptable fake provider. Each call pops the next response from the queue.
struct FakeProvider {
    responses: std::sync::Mutex<Vec<FakeResponse>>,
}

struct FakeResponse {
    chunks: Vec<Chunk>,
}

impl FakeProvider {
    fn new(responses: Vec<FakeResponse>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
        }
    }
}

#[async_trait]
impl Provider for FakeProvider {
    async fn request(&self, _req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
        let response = {
            let mut queue = self.responses.lock().unwrap();
            if queue.is_empty() {
                anyhow::bail!("FakeProvider: no more responses");
            }
            queue.remove(0)
        };
        for chunk in response.chunks {
            let _ = tx.send(chunk).await;
        }
        Ok(())
    }
}

// --- Helpers ---

async fn run_agent(provider: FakeProvider, prompt: &str) -> Vec<StreamEvent> {
    let tools = Arc::new(ToolRegistry::with_defaults());
    let agent = Agent::new(
        Arc::new(provider),
        tools,
        "fake-model".into(),
        "You are a test assistant.".into(),
    );

    let (tx, mut rx) = mpsc::channel(256);
    let handle = tokio::spawn(async move {
        let mut events = Vec::new();
        while let Some(e) = rx.recv().await {
            events.push(e);
        }
        events
    });

    agent.run(prompt.into(), tx).await.unwrap();
    handle.await.unwrap()
}

fn text_response(text: &str) -> FakeResponse {
    FakeResponse {
        chunks: vec![
            Chunk::Text(text.into()),
            Chunk::Done {
                stop_reason: StopReason::EndTurn,
                content: vec![ContentBlock::Text { text: text.into() }],
            },
        ],
    }
}

fn tool_call_response(id: &str, name: &str, input: serde_json::Value) -> FakeResponse {
    FakeResponse {
        chunks: vec![
            Chunk::ToolUse {
                id: id.into(),
                name: name.into(),
                input: input.clone(),
            },
            Chunk::Done {
                stop_reason: StopReason::ToolUse,
                content: vec![ContentBlock::ToolUse {
                    id: id.into(),
                    name: name.into(),
                    input,
                }],
            },
        ],
    }
}

// --- Tests ---

/// Full round trip: agent calls bash tool, gets result, then responds.
#[tokio::test]
async fn bash_tool_round_trip() {
    let provider = FakeProvider::new(vec![
        tool_call_response("t1", "bash", json!({"command": "echo hello world"})),
        text_response("The command output was: hello world"),
    ]);

    let events = run_agent(provider, "run echo hello world").await;

    let tool_use = events
        .iter()
        .find(|e| matches!(e, StreamEvent::ToolUse { name, .. } if name == "bash"));
    assert!(tool_use.is_some(), "expected ToolUse event");

    let tool_result = events.iter().find(
        |e| matches!(e, StreamEvent::ToolResult { tool_use_id, is_error, .. } if tool_use_id == "t1" && !is_error),
    );
    assert!(tool_result.is_some(), "expected successful ToolResult");

    if let Some(StreamEvent::ToolResult { content, .. }) = tool_result {
        assert!(
            content.contains("hello world"),
            "tool result should contain command output, got: {content}"
        );
    }

    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::Text { text } if text.contains("hello world")))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::Done { reason } if reason == "end_turn"))
    );
}

/// Write a file then read it back in the same agent session.
#[tokio::test]
async fn write_then_read_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let path_str = file_path.to_str().unwrap();

    let provider = FakeProvider::new(vec![
        tool_call_response(
            "t1",
            "write",
            json!({"file_path": path_str, "content": "line1\nline2\nline3"}),
        ),
        tool_call_response("t2", "read", json!({"file_path": path_str})),
        text_response("File has 3 lines."),
    ]);

    let events = run_agent(provider, "write then read a file").await;

    let write_result = events.iter().find(
        |e| matches!(e, StreamEvent::ToolResult { tool_use_id, is_error, .. } if tool_use_id == "t1" && !is_error),
    );
    assert!(write_result.is_some(), "write should succeed");

    let read_result = events
        .iter()
        .find(|e| matches!(e, StreamEvent::ToolResult { tool_use_id, .. } if tool_use_id == "t2"));
    assert!(read_result.is_some(), "read should succeed");
    if let Some(StreamEvent::ToolResult {
        content, is_error, ..
    }) = read_result
    {
        assert!(!is_error);
        assert!(
            content.contains("line1"),
            "read should return written content"
        );
        assert!(content.contains("line3"));
    }
}

/// Multiple tool calls across iterations.
#[tokio::test]
async fn multi_step_tool_usage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("count.txt");
    let path_str = path.to_str().unwrap();

    let provider = FakeProvider::new(vec![
        tool_call_response(
            "t1",
            "bash",
            json!({"command": format!("echo -n '0' > {path_str}")}),
        ),
        tool_call_response("t2", "read", json!({"file_path": path_str})),
        tool_call_response(
            "t3",
            "write",
            json!({"file_path": path_str, "content": "42"}),
        ),
        text_response("Updated to 42."),
    ]);

    let events = run_agent(provider, "create, read, update a file").await;

    let tool_use_count = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::ToolUse { .. }))
        .count();
    assert_eq!(tool_use_count, 3, "should have 3 tool uses");

    let tool_result_count = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::ToolResult { .. }))
        .count();
    assert_eq!(tool_result_count, 3, "should have 3 tool results");

    let final_content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(final_content, "42");
}

/// Tool error is reported correctly.
#[tokio::test]
async fn tool_error_reported() {
    let provider = FakeProvider::new(vec![
        tool_call_response(
            "t1",
            "read",
            json!({"file_path": "/nonexistent/path/file.txt"}),
        ),
        text_response("The file does not exist."),
    ]);

    let events = run_agent(provider, "read a nonexistent file").await;

    let error_result = events.iter().find(
        |e| matches!(e, StreamEvent::ToolResult { tool_use_id, is_error, .. } if tool_use_id == "t1" && *is_error),
    );
    assert!(error_result.is_some(), "should report tool error");
}

/// Unknown tool name is handled gracefully.
#[tokio::test]
async fn unknown_tool_handled() {
    let provider = FakeProvider::new(vec![
        tool_call_response("t1", "nonexistent_tool", json!({})),
        text_response("Sorry, that tool doesn't exist."),
    ]);

    let events = run_agent(provider, "use a fake tool").await;

    let error_result = events.iter().find(
        |e| matches!(e, StreamEvent::ToolResult { tool_use_id, is_error, .. } if tool_use_id == "t1" && *is_error),
    );
    assert!(error_result.is_some(), "unknown tool should return error");

    if let Some(StreamEvent::ToolResult { content, .. }) = error_result {
        assert!(content.contains("Unknown tool"));
    }
}

/// Text-only response with no tool calls.
#[tokio::test]
async fn text_only_no_tools() {
    let provider = FakeProvider::new(vec![text_response("Just a simple answer.")]);

    let events = run_agent(provider, "hello").await;

    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::Text { text } if text == "Just a simple answer."))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::Done { reason } if reason == "end_turn"))
    );

    assert!(
        !events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolUse { .. }))
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolResult { .. }))
    );
}
