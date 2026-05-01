"""
Qwen3.5 2B 矿物知识问答助手 — Unsloth QLoRA 微调脚本
适配 6GB VRAM (RTX 3060)
"""

import torch
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import os

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "unsloth/Qwen3.5-2B"
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
USE_4BIT = True
OUTPUT_DIR = "D:\\OneDrive\\Desktop\\train\\mineral_lora_model"
DATA_DIR = "D:\\OneDrive\\Desktop\\train\\data"

# ============================================================
# ChatML 格式模板 (Qwen3.5 使用 ChatML)
# ============================================================
CHAT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

def format_chatml(examples):
    """将 Alpaca 格式数据转为 ChatML 文本"""
    texts = []
    for i in range(len(examples["instruction"])):
        system = examples.get("system", [""] * len(examples["instruction"]))
        if isinstance(system, list):
            sys = system[i] if i < len(system) else ""
        else:
            sys = system

        inst = examples["instruction"][i]
        inp = examples.get("input", [""] * len(examples["instruction"]))
        if isinstance(inp, list):
            user_input = inp[i] if i < len(inp) else ""
        else:
            user_input = inp

        out = examples["output"][i]

        # 如果有 input 字段，拼接到 instruction 后面
        if user_input and user_input.strip():
            full_instruction = f"{inst}\n{user_input}"
        else:
            full_instruction = inst

        text = CHAT_TEMPLATE.format(
            system=sys,
            instruction=full_instruction,
            output=out,
        )
        texts.append(text)
    return {"text": texts}


# ============================================================
# 1. 加载模型 (4-bit QLoRA)
# ============================================================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,                      # auto-detect
    load_in_4bit=USE_4BIT,
    # RTX 3060 不支持 bf16, 使用 fp16
    # token 可传环境变量 HF_TOKEN，本地缓存过则无需再传
)

# ============================================================
# 2. 添加 LoRA 适配器
# ============================================================
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 节省显存
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# ============================================================
# 3. 加载并格式化数据集
# ============================================================
print("Loading dataset...")
dataset = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "train.json"),
    "eval": os.path.join(DATA_DIR, "eval.json"),
})
dataset = dataset.map(format_chatml, batched=True, remove_columns=dataset["train"].column_names)

print(f"Train samples: {len(dataset['train'])}")
print(f"Eval samples:  {len(dataset['eval'])}")
print(f"\nSample text (first 300 chars):\n{dataset['train'][0]['text'][:300]}...")

# ============================================================
# 4. 训练配置
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,        # effective batch_size = 2 * 4 = 8
    warmup_steps=5,
    max_steps=120,                         # 可调整
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),     # RTX 3060 用 fp16
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    save_steps=60,
    eval_strategy="steps",
    eval_steps=60,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",                      # 不需要 wandb
    run_name="mineral_lora",
    # 以下关键参数控制显存
    dataloader_num_workers=0,             # Windows 下设为 0 避免多进程问题
    gradient_checkpointing=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# ============================================================
# 5. 训练
# ============================================================
print("\nStarting training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=1,
    packing=False,                         # 短问答不需要 packing
)

trainer.train()

# ============================================================
# 6. 保存模型
# ============================================================
print("\nSaving model...")

# 保存 LoRA 适配器
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))

# 保存完整模型（合并 LoRA 到基座，16-bit 用于推理）
model.save_pretrained_merged(os.path.join(OUTPUT_DIR, "merged_16bit"), tokenizer, save_method="merged_16bit")

# 保存 GGUF 格式（用于 Ollama / llama.cpp 部署）
model.save_pretrained_gguf(
    os.path.join(OUTPUT_DIR, "gguf"),
    tokenizer,
    quantization_method="q4_k_m",  # 4-bit 量化，适合 6GB 显卡推理
)

print(f"\nDone! Model saved to: {OUTPUT_DIR}")
print("  - lora_adapter/  : LoRA weights (可继续训练)")
print("  - merged_16bit/  : 合并后的 16-bit 完整模型")
print("  - gguf/          : GGUF Q4_K_M 量化 (用于 Ollama)")
