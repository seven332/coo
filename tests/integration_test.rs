use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::mpsc;

use coo::agent::Agent;
use coo::message::{ContentBlock, StreamEvent};
use coo::provider::{Chunk, Provider, Request, StopReason};
use coo::tools::ToolRegistry;

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

fn has_assistant_with_text(events: &[StreamEvent], expected: &str) -> bool {
    events.iter().any(|e| {
        if let StreamEvent::Assistant { message, .. } = e {
            message
                .get("content")
                .and_then(|c| c.as_array())
                .is_some_and(|blocks| {
                    blocks.iter().any(|b| {
                        b.get("type").and_then(|t| t.as_str()) == Some("text")
                            && b.get("text")
                                .and_then(|t| t.as_str())
                                .is_some_and(|t| t.contains(expected))
                    })
                })
        } else {
            false
        }
    })
}

fn has_result(events: &[StreamEvent], expected_subtype: &str) -> bool {
    events
        .iter()
        .any(|e| matches!(e, StreamEvent::Result { subtype, .. } if subtype == expected_subtype))
}

fn has_user_with_tool_result(events: &[StreamEvent], tool_use_id: &str) -> bool {
    events.iter().any(|e| {
        if let StreamEvent::User { message, .. } = e {
            message.content.iter().any(|c| {
                matches!(c, ContentBlock::ToolResult { tool_use_id: id, .. } if id == tool_use_id)
            })
        } else {
            false
        }
    })
}

fn get_tool_result_content(events: &[StreamEvent], target_id: &str) -> Option<(String, bool)> {
    for e in events {
        if let StreamEvent::User { message, .. } = e {
            for c in &message.content {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } = c
                {
                    if tool_use_id == target_id {
                        let text = content
                            .iter()
                            .map(|tc| match tc {
                                coo::message::ToolResultContent::Text { text } => text.as_str(),
                                _ => panic!("expected text"),
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        return Some((text, is_error.unwrap_or(false)));
                    }
                }
            }
        }
    }
    None
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

    assert!(has_user_with_tool_result(&events, "t1"));

    let (content, is_error) = get_tool_result_content(&events, "t1").unwrap();
    assert!(!is_error);
    assert!(
        content.contains("hello world"),
        "tool result should contain command output, got: {content}"
    );

    assert!(has_assistant_with_text(&events, "hello world"));
    assert!(has_result(&events, "success"));
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

    let (_, write_err) = get_tool_result_content(&events, "t1").unwrap();
    assert!(!write_err, "write should succeed");

    let (read_content, read_err) = get_tool_result_content(&events, "t2").unwrap();
    assert!(!read_err);
    assert!(
        read_content.contains("line1"),
        "read should return written content"
    );
    assert!(read_content.contains("line3"));
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

    // Should have assistant messages for each iteration.
    let assistant_count = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::Assistant { .. }))
        .count();
    assert_eq!(assistant_count, 4, "should have 4 assistant messages");

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

    let (_, is_error) = get_tool_result_content(&events, "t1").unwrap();
    assert!(is_error, "should report tool error");
}

/// Unknown tool name is handled gracefully.
#[tokio::test]
async fn unknown_tool_handled() {
    let provider = FakeProvider::new(vec![
        tool_call_response("t1", "nonexistent_tool", json!({})),
        text_response("Sorry, that tool doesn't exist."),
    ]);

    let events = run_agent(provider, "use a fake tool").await;

    let (content, is_error) = get_tool_result_content(&events, "t1").unwrap();
    assert!(is_error, "unknown tool should return error");
    assert!(content.contains("Unknown tool"));
}

/// Text-only response with no tool calls.
#[tokio::test]
async fn text_only_no_tools() {
    let provider = FakeProvider::new(vec![text_response("Just a simple answer.")]);

    let events = run_agent(provider, "hello").await;

    assert!(has_assistant_with_text(&events, "Just a simple answer."));
    assert!(has_result(&events, "success"));

    // No tool result user messages (only the initial prompt user message).
    assert!(!events.iter().any(|e| {
        if let StreamEvent::User { message, .. } = e {
            message
                .content
                .iter()
                .any(|c| matches!(c, ContentBlock::ToolResult { .. }))
        } else {
            false
        }
    }));
}
