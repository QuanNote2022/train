"""
Qwen3.5-2B 统一多模态训练 — 文本问答 + 图片检测(bbox)
使用 Unsloth FastVisionModel, QLoRA, 适配 6GB VRAM

数据:
  - 文本问答 (180条): 矿物知识 QA + mineralContext + 外观鉴定
  - 图片检测 (300条): 20种矿物 x 15张/种, 输出 JSON bbox

输出格式与后端对齐:
  - ChatService.chatWithOllama()   ← 文本问答
  - MineralService.detectMineral() ← 图片检测 (JSON + bbox)
"""

import torch
from unsloth import FastVisionModel
from datasets import load_dataset, concatenate_datasets, Dataset, Image
from transformers import TrainingArguments
from trl import SFTTrainer
import os, json, random
from collections import defaultdict

random.seed(42)
torch.manual_seed(42)

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen3.5-2B"
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 16
USE_4BIT = True
OUTPUT_DIR = "D:\\OneDrive\\Desktop\\train\\mineral_unified_model"
DATA_DIR = "D:\\OneDrive\\Desktop\\train\\data"

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
# 1. 准备文本问答数据 (ChatML格式)
# ============================================================
print("Preparing text Q&A data...")
text_files = {
    "train": os.path.join(DATA_DIR, "train.json"),
    "eval": os.path.join(DATA_DIR, "eval.json"),
}

text_dataset = load_dataset("json", data_files=text_files)
# text_dataset has "train" and "eval" splits

CHAT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

def format_chatml(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        sys = examples["system"][i]
        inst = examples["instruction"][i]
        out = examples["output"][i]
        texts.append(CHAT_TEMPLATE.format(system=sys, instruction=inst, output=out))
    return {"text": texts}

text_dataset = text_dataset.map(format_chatml, batched=True,
    remove_columns=["system", "instruction", "input", "output"])
print(f"  Text QA train: {len(text_dataset['train'])} eval: {len(text_dataset['eval'])}")

# ============================================================
# 2. 准备图片检测数据
# ============================================================
print("Preparing image detection data...")

SELECTED = {
    "quartz": "石英", "amethyst": "紫水晶", "pyrite": "黄铁矿",
    "calcite": "方解石", "fluorite": "萤石", "hematite": "赤铁矿",
    "magnetite": "磁铁矿", "malachite": "孔雀石", "azurite": "蓝铜矿",
    "gypsum": "石膏", "galena": "方铅矿", "cinnabar": "辰砂",
    "gold": "自然金", "copper": "自然铜", "sulfur": "硫磺",
    "beryl": "绿柱石", "corundum": "刚玉", "topaz": "黄玉",
    "tourmaline": "电气石", "sphalerite": "闪锌矿",
}

# 每种矿物的基本中文描述 (用于数据集 description 为空时填充)
MINERAL_BASE_DESC = {
    "石英": "石英是地壳中含量第二多的矿物，化学成分SiO2，硬度7，玻璃光泽，常见六方柱状晶体。",
    "紫水晶": "紫水晶是石英的紫色变种，因铁离子和天然辐射形成色心而呈紫色，硬度7，玻璃光泽。",
    "黄铁矿": "黄铁矿化学成分为FeS2，金黄色金属光泽，硬度6-6.5，常形成立方体晶体，俗称愚人金。",
    "方解石": "方解石化学成分为CaCO3，硬度3，具有菱面体解理和双折射现象，遇酸冒泡。",
    "萤石": "萤石化合成分CaF2，硬度4，颜色丰富多样，紫外线下可发荧光，八面体解理。",
    "赤铁矿": "赤铁矿化学成分为Fe2O3，条痕樱红色，具有金属至半金属光泽，重要的铁矿石。",
    "磁铁矿": "磁铁矿化学成分为Fe3O4，铁黑色，具有强磁性，是自然界磁性最强的矿物。",
    "孔雀石": "孔雀石是铜碳酸盐矿物，翠绿色，常与蓝铜矿共生，硬度3.5-4，用于矿物颜料和饰品。",
    "蓝铜矿": "蓝铜矿是铜碳酸盐矿物，深蓝色，常与孔雀石共生，硬度3.5-4，用作蓝色矿物颜料。",
    "石膏": "石膏化学成分为CaSO4·2H2O，硬度2，常见纤维状或片状集合体，广泛用于建筑和医药。",
    "方铅矿": "方铅矿化学成分为PbS，铅灰色金属光泽，比重极大(7.5)，具有完美的立方体解理。",
    "辰砂": "辰砂化学成分为HgS，鲜红色，条痕红色，硬度2-2.5，是主要的汞矿石，又称朱砂。",
    "自然金": "自然金化学成分为Au，金黄色金属光泽，硬度2.5-3，具极好的延展性，化学性质稳定。",
    "自然铜": "自然铜化学成分为Cu，铜红色金属光泽，硬度2.5-3，具延展性，氧化后表面变暗。",
    "硫磺": "自然硫化学成分为S，柠檬黄色，油脂至金刚光泽，硬度1.5-2.5，产于火山喷气孔。",
    "绿柱石": "绿柱石化合成Be3Al2Si6O18，硬度7.5-8，祖母绿和海蓝宝石都是绿柱石的变种。",
    "刚玉": "刚玉化学成分为Al2O3，硬度9，红宝石和蓝宝石都是刚玉的颜色变种。",
    "黄玉": "黄玉化学成分为Al2SiO4(F,OH)2，硬度8，斜方晶系，常见柱状晶体，又称托帕石。",
    "电气石": "电气石化学成分复杂，含硼环状硅酸盐，硬度7-7.5，具有热电性和压电性，又称碧玺。",
    "闪锌矿": "闪锌矿化学成分为ZnS，硬度3.5-4，油脂至金刚光泽，是锌的最主要矿石矿物。",
}

DETECTION_PROMPTS = [
    "请识别图片中的矿物，给出名称和位置框。",
    "这张图片里有什么矿物？请用JSON格式返回，包含名称、置信度和边界框。",
    "分析这张矿物图片，输出识别结果和bbox坐标。",
]

# 使用流式加载+缓存来加速
print("  Loading mineral images (streaming)...")
vlds = load_dataset('Nech-C/mineralimage5K-98', split='train', streaming=True)
class_names = vlds.features['name'].names
target_ids = {i for i, name in enumerate(class_names) if name in SELECTED}

groups = defaultdict(list)
for item in vlds:
    if item['name'] not in target_ids:
        continue
    mineral = class_names[item['name']]
    if len(groups[mineral]) < 15:
        groups[mineral].append(item)
    if all(len(v) >= 15 for v in groups.values()):
        break

# 生成图片检测样本
vl_images = []
vl_texts = []

for mineral in SELECTED:
    cn_name = SELECTED[mineral]
    for item in groups[mineral]:
        boxes = item.get('mineral_boxes', [])
        if boxes:
            bbox = boxes[0]['box']
            bbox_pixel = [int(bbox[0]*1000), int(bbox[1]*1000),
                          int(bbox[2]*1000), int(bbox[3]*1000)]
        else:
            bbox_pixel = [0, 0, 1000, 1000]

        # 用数据集描述或预设的中文描述
        desc = item.get('description', '').strip()
        if not desc or len(desc) < 10:
            desc = MINERAL_BASE_DESC.get(cn_name, f"{cn_name}（{mineral}）")

        response = json.dumps([{
            "label": cn_name,
            "confidence": 0.90,
            "bbox": bbox_pixel,
            "mineralInfo": {
                "name": cn_name,
                "formula": "",
                "hardness": "",
                "luster": "",
                "color": "",
                "description": desc[:150],
            }
        }], ensure_ascii=False, indent=2)

        prompt = random.choice(DETECTION_PROMPTS)
        text = CHAT_TEMPLATE.format(
            system=DETECTION_SYSTEM,
            instruction=prompt,
            output=response,
        )

        vl_images.append(item['image'])
        vl_texts.append(text)

print(f"  Image detection samples: {len(vl_texts)}")

# 创建 VL dataset (带 image 列)
vl_dataset = Dataset.from_dict({
    "image": vl_images,
    "text": vl_texts,
}).cast_column("image", Image())

# 划分 VL 训练/验证
vl_splits = vl_dataset.train_test_split(test_size=0.1, seed=42)
vl_train = vl_splits["train"]
vl_eval = vl_splits["test"]
print(f"  VL train: {len(vl_train)} eval: {len(vl_eval)}")

# ============================================================
# 3. 合并文本和图片数据集
# ============================================================
print("Merging text + image datasets...")
# 文本 QA 数据集添加空的 image 列 (让 Qwen3.5 按纯文本处理)
text_train_noimg = text_dataset["train"].add_column("image", [None] * len(text_dataset["train"]))
text_eval_noimg = text_dataset["eval"].add_column("image", [None] * len(text_dataset["eval"]))

# 合并
train_dataset = concatenate_datasets([text_train_noimg, vl_train])
eval_dataset = concatenate_datasets([text_eval_noimg, vl_eval])

print(f"  Combined train: {len(train_dataset)} eval: {len(eval_dataset)}")

# ============================================================
# 4. 加载模型
# ============================================================
print("\nLoading Qwen3.5-2B...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=USE_4BIT,
)

# ============================================================
# 5. 添加 LoRA
# ============================================================
print("Adding LoRA...")
model = FastVisionModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ============================================================
# 6. 训练
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,       # VL 训练 batch=1
    gradient_accumulation_steps=8,       # effective = 8
    warmup_steps=10,
    max_steps=180,                         # ~366样本 / 8 effective batch ≈ 46 steps/epoch, ~4 epochs
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    save_total_limit=2,
    remove_unused_columns=False,
)

print("\nStarting unified training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
)

trainer.train()

# ============================================================
# 7. 保存
# ============================================================
print("\nSaving model...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
model.save_pretrained_merged(
    os.path.join(OUTPUT_DIR, "merged_16bit"),
    tokenizer,
    save_method="merged_16bit",
)
model.save_pretrained_gguf(
    os.path.join(OUTPUT_DIR, "gguf"),
    tokenizer,
    quantization_method="q4_k_m",
)

print(f"\nDone! Model saved to: {OUTPUT_DIR}")
