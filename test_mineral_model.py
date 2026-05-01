"""
Qwen3.5-2B 矿物识别助手 — 推理测试
支持: 文本问答 / mineralContext(带矿物上下文) / 图片检测
"""
import torch
from unsloth import FastVisionModel
from PIL import Image
import sys, os, json

MODEL_PATH = "D:\\OneDrive\\Desktop\\train\\mineral_unified_model\\merged_16bit"
LORA_PATH = "D:\\OneDrive\\Desktop\\train\\mineral_unified_model\\lora_adapter"
USE_LORA = os.path.exists(LORA_PATH)

SYSTEM_PROMPT = (
    "你是一个专业的矿物识别助手，专门帮助用户了解矿物相关知识。"
    "请用简洁、准确、友好的中文回答用户的问题。\n\n"
    "如果用户的问题超出你的知识范围，请诚实地告知，不要编造信息。"
)

DETECTION_SYSTEM = (
    "你是一个专业的矿物识别专家。请仔细分析图片，识别图片中的矿物。"
    "请以JSON格式返回识别结果，包含矿物名称、置信度和边界框(bbox)。"
)

# ============================================================
print("Loading model...")
if os.path.exists(MODEL_PATH):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_PATH, max_seq_length=1024, dtype=None, load_in_4bit=True,
    )
    print(f"Loaded from {MODEL_PATH}")
elif USE_LORA:
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="Qwen/Qwen3.5-2B", max_seq_length=1024, dtype=None, load_in_4bit=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print(f"Loaded base + LoRA from {LORA_PATH}")
else:
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="Qwen/Qwen3.5-2B", max_seq_length=1024, dtype=None, load_in_4bit=True,
    )
    print("Loaded base model (no fine-tune)")

FastVisionModel.for_inference(model)

# ============================================================
def ask(question: str, mineral_context: str = None, image: Image.Image = None) -> str:
    """调用 Qwen3.5 多模态模型"""
    # 构建 system prompt
    if image is not None:
        system_content = DETECTION_SYSTEM
    elif mineral_context:
        system_content = SYSTEM_PROMPT + "\n\n=== 当前矿物信息 ===\n" + mineral_context + "\n=== 当前矿物信息结束 ==="
    else:
        system_content = SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50,
            repetition_penalty=1.1, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()


# ============================================================
# 测试用例
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  Qwen3.5-2B 矿物识别助手 — 推理测试")
    print("=" * 60)

    # ── 一、文本问答 ──
    print("\n" + "─" * 50)
    print("  一、文本问答")
    print("─" * 50)
    questions = [
        "什么是莫氏硬度？",
        "石英和方解石怎么区分？",
        "请介绍一下黄铁矿。",
        "黄金为什么是黄色的？",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)[:300]}")

    # ── 二、矿物上下文问答 ──
    print("\n" + "─" * 50)
    print("  二、mineralContext (模拟后端识别后咨询)")
    print("─" * 50)
    ctx_q = [
        ("这个矿物有什么用途？", "名称: 磁铁矿\n化学式: Fe3O4\n硬度: 5.5-6.5\n光泽: 金属光泽\n颜色: 铁黑色\n用途: 铁矿石、磁性材料"),
        ("它有毒吗？怎么安全收藏？", "名称: 辰砂\n化学式: HgS\n硬度: 2-2.5\n颜色: 鲜红色\n描述: 辰砂含汞，加热释放汞蒸气"),
    ]
    for q, ctx in ctx_q:
        print(f"\n[矿物信息]: {ctx[:60]}...")
        print(f"Q: {q}")
        print(f"A: {ask(q, mineral_context=ctx)[:300]}")

    # ── 三、图片检测 ──
    print("\n" + "─" * 50)
    print("  三、图片检测 (如果有测试图片)")
    print("─" * 50)
    test_img_path = "D:\\OneDrive\\Desktop\\train\\test_mineral.jpg"
    if os.path.exists(test_img_path):
        img = Image.open(test_img_path).convert("RGB")
        print(f"  图片: {test_img_path} ({img.size})")
        result = ask("请识别图片中的矿物。", image=img)
        print(f"  结果: {result[:500]}")
    else:
        print("  (无测试图片，跳过。放一张 test_mineral.jpg 到 train 目录即可测试)")

    # ── 交互模式 ──
    if "--interactive" in sys.argv:
        print("\n" + "=" * 60)
        print("  交互模式 (输入 quit 退出)")
        print("=" * 60)
        while True:
            try:
                q = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q.startswith("@"):
                path = q[1:].strip()
                try:
                    img = Image.open(path).convert("RGB")
                    print(f"  [已加载图片: {img.size}]")
                    det_q = input("  (按回车用默认检测提示词): ").strip()
                    if not det_q:
                        det_q = "请识别图片中的矿物，给出名称和位置框。"
                    print("助手: ", end="", flush=True)
                    print(ask(det_q, image=img))
                except Exception as e:
                    print(f"  加载图片失败: {e}")
                continue
            if not q:
                continue
            print("助手: ", end="", flush=True)
            print(ask(q))


if __name__ == "__main__":
    main()
