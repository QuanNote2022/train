"""
重新构建数据集：
1. 更新所有 system prompt 为后端使用的版本
2. 合并 mineralContext 场景的额外数据
3. 重新划分 train/eval
"""
import json, random, os, copy

random.seed(42)

# 后端实际使用的 System Prompt
BACKEND_SYSTEM_PROMPT = (
    "你是一个专业的矿物识别助手，专门帮助用户了解矿物相关知识。"
    "请用简洁、准确、友好的中文回答用户的问题。\n\n"
    "如果用户的问题超出你的知识范围，请诚实地告知，不要编造信息。"
)

DATA_DIR = r"D:\OneDrive\Desktop\train\data"

# 1. 加载原有数据
with open(os.path.join(DATA_DIR, "train.json"), "r", encoding="utf-8") as f:
    original_data = json.load(f)

# 2. 更新所有 system prompt
for item in original_data:
    item["system"] = BACKEND_SYSTEM_PROMPT

print(f"Original samples: {len(original_data)}")

# 3. 加载额外数据
with open(os.path.join(DATA_DIR, "extra_samples.json"), "r", encoding="utf-8") as f:
    extra_data = json.load(f)

print(f"Extra samples: {len(extra_data)}")

# 4. 合并
all_data = original_data + extra_data
random.shuffle(all_data)

# 5. 划分 (90/10)
split_idx = int(len(all_data) * 0.9)
train_data = all_data[:split_idx]
eval_data = all_data[split_idx:]

# 6. 保存
with open(os.path.join(DATA_DIR, "train.json"), "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(os.path.join(DATA_DIR, "eval.json"), "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)

print(f"\nFinal dataset:")
print(f"  Train: {len(train_data)} samples")
print(f"  Eval:  {len(eval_data)} samples")
print(f"  Total: {len(all_data)} samples")

# 统计包含 mineralContext 的样本数
ctx_count = sum(1 for item in all_data if "当前矿物信息" in (item.get("input", "") or ""))
print(f"  With mineralContext: {ctx_count} samples")

# 验证 system prompt 一致性
sample_prompts = set(item.get("system", "")[:30] for item in all_data)
print(f"  Unique system prompts: {len(sample_prompts)}")
print("Done!")
