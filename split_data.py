import json, random, os

random.seed(42)

with open(r"D:\OneDrive\Desktop\train\data\train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

random.shuffle(data)
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
eval_data = data[split_idx:]

os.makedirs(r"D:\OneDrive\Desktop\train\data", exist_ok=True)

with open(r"D:\OneDrive\Desktop\train\data\train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(r"D:\OneDrive\Desktop\train\data\eval.json", "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train_data)} samples")
print(f"Eval:  {len(eval_data)} samples")
print(f"Total: {len(data)} samples")
print("Done!")
