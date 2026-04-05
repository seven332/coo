use async_trait::async_trait;
use rand::Rng as _;
use serde_json::json;
use tokio::sync::mpsc;

use super::{Chunk, Provider, Request, StopReason};
use crate::message::ContentBlock;

const COOS: &[&str] = &[
    // Chinese
    "咕咕~",
    "咕咕咕！",
    "咕？",
    "咕噜噜~",
    "咕咕咕咕咕咕咕！！！",
    // English
    "coo~",
    "coo coo",
    "coooooo",
    "brrrr",
    "coo coo coo!",
    // Japanese
    "ぽっぽ〜",
    "くるっくー！",
    "ポッポ",
    // Korean
    "구구~",
    "구구구구!",
    // French
    "roucou~",
    "roucou roucou !",
    // German
    "gurr~",
    "gurr gurr!",
    // Spanish
    "curu~",
    "¡cucurrucucú!",
    // Russian
    "гуль-гуль~",
    "курлык!",
    // Italian
    "tu tu~",
    "tu tu ru!",
    // Thai
    "กุ๊กกรู~",
    // Arabic
    "هدهد~",
    // Turkish
    "gugu~",
    // Actions
    "(点头走路)",
    "(啄食面包屑)",
    "(歪头看你)",
    "(抖翅膀)",
    "(在广场散步)",
    "(head bobbing intensifies)",
    "(pecks at breadcrumbs)",
    "(struts confidently)",
    "(ポッポと歩く)",
    "(머리를 까딱까딱)",
];

const THOUGHTS: &[&str] = &[
    "那块面包看起来不错...",
    "要不要飞到那个屋顶上...",
    "这个广场是我的地盘",
    "人类在说什么？算了继续走",
    "面包面包面包",
    "Should I land on that statue? ...yes.",
    "あのパン屑...食べなきゃ！",
    "Soll ich auf die Statue fliegen?",
    "Cette place est parfaite pour se promener",
    "그 빵 부스러기... 먹어야 해!",
    "¿Debería caminar por aquí? ...sí.",
    "Может сесть на памятник? ...да.",
];

const BASH_COMMANDS: &[&str] = &[
    "echo 咕咕",
    "date",
    "whoami",
    "ls",
    "pwd",
    "echo coo~",
    "cat /dev/null",
    "echo 🐦",
    "uname -a",
    "echo coo | rev",
];

const READ_FILES: &[&str] = &[
    "/etc/hostname",
    "/etc/shells",
    "/tmp/.coo",
    "Cargo.toml",
    "README.md",
];

pub struct CooProvider;

/// All random decisions made before any await.
struct CooPlan {
    thought: Option<String>,
    text: String,
    tool_call: Option<ToolCall>,
}

struct ToolCall {
    id: String,
    name: String,
    input: serde_json::Value,
}

fn plan(req: &Request) -> CooPlan {
    let mut rng = rand::rng();

    // Build coo text.
    let coo_count = rng.random_range(1..=5);
    let mut text = String::new();
    for _ in 0..coo_count {
        let coo = COOS[rng.random_range(0..COOS.len())];
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(coo);
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
        let id = format!("coo_{}", rng.random_range(1000..9999));
        Some(ToolCall {
            id,
            name: tool.name.clone(),
            input,
        })
    } else {
        None
    };

    CooPlan {
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
            let content = COOS[rng.random_range(0..COOS.len())];
            json!({
                "file_path": format!("/tmp/coo_{}.txt", rng.random_range(0..100)),
                "content": content,
            })
        }
        _ => json!({}),
    }
}

#[async_trait]
impl Provider for CooProvider {
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
