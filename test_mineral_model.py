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

SYSTEM_PROMPT = "你是一位知识渊博的矿物学家和科普达人，善于用通俗易懂的语言向大众科普矿物学知识。你的回答要准确、生动、易于理解。"

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
def ask(question: str) -> str:
    """向矿物助手提问"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
    "什么是莫氏硬度？",
    "石英和方解石怎么区分？",
    "金刚石和石墨有什么区别？",
    "请介绍一下黄铁矿。",
    "红宝石和蓝宝石有什么关系？",
    "我们日常生活中哪些东西来自矿物？",
    "请详细介绍萤石这种矿物。",
    "和田玉是什么矿物组成的？",
    "怎么样用简单的方法鉴别矿物？",
    "黄金为什么是黄色的？",
]


def run_tests():
    print("\n" + "=" * 60)
    print("  矿物知识问答助手 — 推理测试")
    print("=" * 60 + "\n")

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'─' * 50}")
        print(f"Q{i}: {q}")
        print(f"{'─' * 50}")
        answer = ask(q)
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
