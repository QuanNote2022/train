import json, os

DATA_DIR = r"D:\OneDrive\Desktop\train\data"

for fname in ["train.json", "eval.json"]:
    fpath = os.path.join(DATA_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    fixed = 0
    for item in data:
        if '系统' in item:  # 系统
            print(f"[{fname}] removing '系统' key: {item.get('系统', '')[:50]}...")
            del item['系统']
            fixed += 1
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[{fname}] checked {len(data)}, fixed {fixed}")
