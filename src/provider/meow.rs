use async_trait::async_trait;
use rand::Rng as _;
use serde_json::json;
use tokio::sync::mpsc;

use super::{Chunk, Provider, Request, StopReason};
use crate::message::ContentBlock;

const MEOWS: &[&str] = &[
    // Chinese
    "喵~",
    "喵喵喵！",
    "喵？",
    "喵呜~",
    "喵喵喵喵喵喵喵！！！",
    // English
    "meow~",
    "meoooow",
    "prrrrrr",
    "hiss!",
    "mew mew",
    // Japanese
    "にゃ〜",
    "にゃんにゃん！",
    "nyaa~",
    "ニャー",
    // Korean
    "야옹~",
    "냐옹냐옹！",
    // French
    "miaou~",
    "miaou miaou !",
    // German
    "miau~",
    "miau miau!",
    // Spanish
    "miau~",
    "¡miau miau!",
    // Russian
    "мяу~",
    "мяу мяу!",
    // Italian
    "miao~",
    "miao miao!",
    // Thai
    "เหมียว~",
    // Arabic
    "مياو~",
    // Turkish
    "miyav~",
    // Actions (multilingual)
    "(蹭蹭)",
    "(翻肚皮)",
    "(打翻水杯)",
    "(盯着你看)",
    "(疯狂跑酷)",
    "(purring intensifies)",
    "(knocks things off table)",
    "(sits in box)",
    "(ゴロゴロ)",
    "(꾹꾹이)",
];

const THOUGHTS: &[&str] = &[
    "让我想想...要不要跳上桌子...",
    "这个纸箱看起来很好坐",
    "红点！红点在哪！",
    "主人在说什么？算了不重要",
    "鱼鱼鱼鱼鱼",
    "Should I knock this off the table? ...yes.",
    "あの赤い点...捕まえなきゃ！",
    "Soll ich auf den Tisch springen?",
    "Ce carton a l'air parfait pour s'asseoir",
    "그 빨간 점... 잡아야 해!",
    "¿Debería tumbarme aquí? ...sí.",
    "Может прыгнуть на стол? ...да.",
];

const BASH_COMMANDS: &[&str] = &[
    "echo 喵",
    "date",
    "whoami",
    "ls",
    "pwd",
    "echo purrrr",
    "cat /dev/null",
    "echo 🐱",
    "uname -a",
    "echo meow | rev",
];

const READ_FILES: &[&str] = &[
    "/etc/hostname",
    "/etc/shells",
    "/tmp/.meow",
    "Cargo.toml",
    "README.md",
];

pub struct MeowProvider;

/// All random decisions made before any await.
struct MeowPlan {
    thought: Option<String>,
    text: String,
    tool_call: Option<ToolCall>,
}

struct ToolCall {
    id: String,
    name: String,
    input: serde_json::Value,
}

fn plan(req: &Request) -> MeowPlan {
    let mut rng = rand::rng();

    // Build meow text.
    let meow_count = rng.random_range(1..=5);
    let mut text = String::new();
    for _ in 0..meow_count {
        let meow = MEOWS[rng.random_range(0..MEOWS.len())];
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(meow);
    }

    // Maybe think.
    let thought = if rng.random_bool(0.3) {
        Some(THOUGHTS[rng.random_range(0..THOUGHTS.len())].to_string())
    } else {
        None
    };

    // Maybe call a tool.
    let tool_call = if rng.random_bool(0.6) && !req.tools.is_empty() {
        let tool = &req.tools[rng.random_range(0..req.tools.len())];
        let input = random_input_for_tool(&tool.name, &mut rng);
        let id = format!("meow_{}", rng.random_range(1000..9999));
        Some(ToolCall {
            id,
            name: tool.name.clone(),
            input,
        })
    } else {
        None
    };

    MeowPlan {
        thought,
        text,
        tool_call,
    }
}

fn random_input_for_tool(name: &str, rng: &mut impl rand::Rng) -> serde_json::Value {
    match name {
        "bash" => {
            json!({"command": BASH_COMMANDS[rng.random_range(0..BASH_COMMANDS.len())]})
        }
        "read" => {
            json!({"file_path": READ_FILES[rng.random_range(0..READ_FILES.len())]})
        }
        "write" => {
            let content = MEOWS[rng.random_range(0..MEOWS.len())];
            json!({
                "file_path": format!("/tmp/meow_{}.txt", rng.random_range(0..100)),
                "content": content,
            })
        }
        _ => json!({}),
    }
}

#[async_trait]
impl Provider for MeowProvider {
    async fn request(&self, req: Request, tx: mpsc::Sender<Chunk>) -> anyhow::Result<()> {
        let plan = plan(&req);

        // Think.
        if let Some(thought) = plan.thought {
            let _ = tx.send(Chunk::Thinking(thought)).await;
        }

        // Stream text character by character.
        for ch in plan.text.chars() {
            let _ = tx.send(Chunk::Text(ch.to_string())).await;
        }

        // Tool call or end.
        if let Some(tc) = plan.tool_call {
            let _ = tx
                .send(Chunk::ToolUse {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    input: tc.input.clone(),
                })
                .await;

            let _ = tx
                .send(Chunk::Done {
                    stop_reason: StopReason::ToolUse,
                    content: vec![
                        ContentBlock::Text { text: plan.text },
                        ContentBlock::ToolUse {
                            id: tc.id,
                            name: tc.name,
                            input: tc.input,
                        },
                    ],
                })
                .await;
        } else {
            let _ = tx
                .send(Chunk::Done {
                    stop_reason: StopReason::EndTurn,
                    content: vec![ContentBlock::Text { text: plan.text }],
                })
                .await;
        }

        Ok(())
    }
}
