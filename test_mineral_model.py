"""
矿物知识问答助手 — 推理测试脚本
加载微调后的模型进行交互式问答
"""

import torch
from unsloth import FastLanguageModel
import sys
import os

MODEL_PATH = "D:\\OneDrive\\Desktop\\train\\mineral_lora_model\\merged_16bit"
# 如果用 LoRA 适配器而非合并后的模型，取消下面一行注释：
# MODEL_PATH = "unsloth/Qwen3.5-2B"
# LORA_PATH = "D:\\OneDrive\\Desktop\\train\\mineral_lora_model\\lora_adapter"

# 与后端 ChatService.buildSystemPrompt() 保持一致
SYSTEM_PROMPT = (
    "你是一个专业的矿物识别助手，专门帮助用户了解矿物相关知识。"
    "请用简洁、准确、友好的中文回答用户的问题。\n\n"
    "如果用户的问题超出你的知识范围，请诚实地告知，不要编造信息。"
)

# ============================================================
# 加载模型
# ============================================================
print("Loading model...")

if os.path.exists(MODEL_PATH):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Loaded merged model from {MODEL_PATH}")
else:
    # 基座 + LoRA 方式加载
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-2B",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        model,
        "D:\\OneDrive\\Desktop\\train\\mineral_lora_model\\lora_adapter"
    )
    print("Loaded base model + LoRA adapter")

FastLanguageModel.for_inference(model)

# ============================================================
# 推理函数
# ============================================================
def ask(question: str, mineral_context: str = None) -> str:
    """
    向矿物助手提问。
    模拟后端 ChatService.chatWithOllama() 的消息构建逻辑:
      SystemMessage(systemPrompt + mineralContext)
      UserMessage(question)
    """
    system_content = SYSTEM_PROMPT
    if mineral_context:
        system_content += "\n\n=== 当前矿物信息 ===\n" + mineral_context + "\n=== 当前矿物信息结束 ==="

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response.strip()


# ============================================================
# 测试用例
# ============================================================
TEST_QUESTIONS = [
    # 基础矿物问答 (无上下文)
    "什么是莫氏硬度？",
    "石英和方解石怎么区分？",
    "金刚石和石墨有什么区别？",
    "请介绍一下黄铁矿。",
    "我们日常生活中哪些东西来自矿物？",
    "黄金为什么是黄色的？",
]

# 模拟后端识别后带 mineralContext 的问答场景
TEST_CONTEXTUAL = [
    {
        "question": "这个矿物有什么用途？",
        "context": "名称: 磁铁矿\n化学式: Fe₃O₄\n硬度: 5.5-6.5\n光泽: 金属至半金属光泽\n颜色: 铁黑色\n产地: 岩浆岩、变质岩、砂矿\n用途: 铁矿石、磁性材料\n描述: 磁铁矿是自然界磁性最强的矿物，古代指南针「司南」即用天然磁铁矿制成。",
    },
    {
        "question": "它有毒吗？怎么安全收藏？",
        "context": "名称: 雄黄\n化学式: As₄S₄\n硬度: 1.5-2\n光泽: 油脂至金刚光泽\n颜色: 橙红色\n产地: 低温热液矿床\n用途: 中药(慎用)、矿物收藏\n描述: 雄黄含砷，具有毒性，加热产生砒霜。与雌黄常共生，称「矿物鸳鸯」。",
    },
    {
        "question": "这个矿物硬度高吗？日常佩戴要注意什么？",
        "context": "名称: 萤石\n化学式: CaF₂\n硬度: 4\n光泽: 玻璃光泽\n颜色: 紫色/绿色/蓝色\n产地: 热液矿床\n用途: 冶金助熔剂、光学镜片\n描述: 萤石颜色丰富，紫外线下发荧光，但硬度低易划伤。",
    },
]


def run_tests():
    print("\n" + "=" * 60)
    print("  矿物识别助手 — 推理测试")
    print("=" * 60 + "\n")

    # 1. 基础问答
    print("─" * 50)
    print("  一、基础矿物问答")
    print("─" * 50)
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\nQ{i}: {q}")
        answer = ask(q)
        print(f"A: {answer}")
        print()

    # 2. 带 mineralContext 的问答 (模拟后端识别后咨询场景)
    print("\n" + "─" * 50)
    print("  二、矿物识别后咨询 (mineralContext)")
    print("─" * 50)
    for i, item in enumerate(TEST_CONTEXTUAL, 1):
        print(f"\n[矿物信息]: {item['context'][:80]}...")
        print(f"Q{i}: {item['question']}")
        answer = ask(item["question"], mineral_context=item["context"])
        print(f"A: {answer}")
        print()


def interactive_mode():
    print("\n" + "=" * 60)
    print("  矿物知识问答助手 — 交互模式")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        print("助手: ", end="", flush=True)
        answer = ask(q)
        print(answer)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        run_tests()
        print("\n" + "=" * 60)
        print("  测试完成！运行 --interactive 进入交互模式")
        print("=" * 60 + "\n")
