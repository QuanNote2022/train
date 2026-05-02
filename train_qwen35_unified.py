"""
Qwen3.5-2B 统一训练 — 文本问答 + 图片检测
Step 1: 训练文本问答 (纯文本 ChatML)
Step 2: 训练图片检测 (图片 + JSON bbox)
QLoRA, 适配 6GB VRAM
"""

import torch
from unsloth import FastVisionModel
from datasets import load_dataset, Dataset, Image
from transformers import TrainingArguments
from trl import SFTTrainer
import os, json, random
from collections import defaultdict

random.seed(42)
torch.manual_seed(42)

MODEL_NAME = "Qwen/Qwen3.5-2B"
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 16
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

def format_chatml(system, instruction, output):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

# ============================================================
# Step 1: 文本问答数据
# ============================================================
print("=" * 50)
print("  Step 1: 准备文本问答数据")
print("=" * 50)

def build_text_dataset(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        texts.append(format_chatml(
            examples["system"][i],
            examples["instruction"][i],
            examples["output"][i],
        ))
    return {"text": texts}

text_ds = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "train.json"),
    "eval": os.path.join(DATA_DIR, "eval.json"),
})
text_ds = text_ds.map(build_text_dataset, batched=True,
    remove_columns=["system", "instruction", "input", "output"])

print(f"  Text QA: train={len(text_ds['train'])}, eval={len(text_ds['eval'])}")

# ============================================================
# Step 2: 图片检测数据
# ============================================================
print("\n" + "=" * 50)
print("  Step 2: 准备图片检测数据")
print("=" * 50)

SELECTED = {
    "quartz": "石英", "amethyst": "紫水晶", "pyrite": "黄铁矿",
    "calcite": "方解石", "fluorite": "萤石", "hematite": "赤铁矿",
    "magnetite": "磁铁矿", "malachite": "孔雀石", "azurite": "蓝铜矿",
    "gypsum": "石膏", "galena": "方铅矿", "cinnabar": "辰砂",
    "gold": "自然金", "copper": "自然铜", "sulfur": "硫磺",
    "beryl": "绿柱石", "corundum": "刚玉", "topaz": "黄玉",
    "tourmaline": "电气石", "sphalerite": "闪锌矿",
}

MINERAL_DESC = {
    "石英": "石英SiO2，硬度7，玻璃光泽，地壳含量第二的矿物。",
    "紫水晶": "紫水晶是石英的紫色变种，铁离子+辐射形成色心致色，硬度7。",
    "黄铁矿": "黄铁矿FeS2，金黄色金属光泽，硬度6-6.5，常呈立方体晶体，俗称愚人金。",
    "方解石": "方解石CaCO3，硬度3，菱面体解理，双折射，遇酸冒泡。",
    "萤石": "萤石CaF2，硬度4，颜色丰富，紫外线下发荧光，八面体解理。",
    "赤铁矿": "赤铁矿Fe2O3，条痕樱红色，金属至半金属光泽，重要铁矿石。",
    "磁铁矿": "磁铁矿Fe3O4，铁黑色，强磁性，自然界磁性最强的矿物。",
    "孔雀石": "孔雀石铜碳酸盐矿物，翠绿色，常与蓝铜矿共生，硬度3.5-4。",
    "蓝铜矿": "蓝铜矿铜碳酸盐矿物，深蓝色，常与孔雀石共生，硬度3.5-4。",
    "石膏": "石膏CaSO4·2H2O，硬度2，纤维状或片状，用于建筑和医药。",
    "方铅矿": "方铅矿PbS，铅灰色金属光泽，比重7.5，立方体解理。",
    "辰砂": "辰砂HgS，鲜红色，条痕红色，硬度2-2.5，主要汞矿石，又称朱砂。",
    "自然金": "自然金Au，金黄色金属光泽，硬度2.5-3，极好延展性。",
    "自然铜": "自然铜Cu，铜红色金属光泽，硬度2.5-3，具延展性。",
    "硫磺": "自然硫S，柠檬黄色，油脂至金刚光泽，硬度1.5-2.5。",
    "绿柱石": "绿柱石Be3Al2Si6O18，硬度7.5-8，祖母绿和海蓝宝石是其变种。",
    "刚玉": "刚玉Al2O3，硬度9，红宝石和蓝宝石是刚玉的颜色变种。",
    "黄玉": "黄玉Al2SiO4(F,OH)2，硬度8，斜方晶系柱状晶体，又称托帕石。",
    "电气石": "电气石含硼环状硅酸盐，硬度7-7.5，有热电性和压电性，又称碧玺。",
    "闪锌矿": "闪锌矿ZnS，硬度3.5-4，油脂至金刚光泽，锌最主要矿石矿物。",
}

DETECTION_PROMPTS = [
    "请识别图片中的矿物，给出名称和位置框。",
    "这张图片里有什么矿物？请用JSON格式返回，包含名称、置信度和边界框。",
    "分析这张矿物图片，输出识别结果和bbox坐标。",
]

print("  Loading mineral images...")
vlds = load_dataset('Nech-C/mineralimage5K-98', split='train', streaming=True)
class_names = vlds.features['name'].names
target_ids = {i for i, name in enumerate(class_names) if name in SELECTED}

groups = defaultdict(list)
for item in vlds:
    if item['name'] not in target_ids: continue
    mineral = class_names[item['name']]
    if len(groups[mineral]) < 15: groups[mineral].append(item)
    if all(len(v) >= 15 for v in groups.values()): break

img_images, img_texts = [], []
for mineral in SELECTED:
    cn = SELECTED[mineral]
    for item in groups[mineral]:
        boxes = item.get('mineral_boxes', [])
        b = boxes[0]['box'] if boxes else [0, 0, 1, 1]
        bbox_px = [int(b[0]*1000), int(b[1]*1000), int(b[2]*1000), int(b[3]*1000)]
        desc = item.get('description', '').strip()
        if len(desc) < 10: desc = MINERAL_DESC.get(cn, cn)

        response = json.dumps([{
            "label": cn, "confidence": 0.90, "bbox": bbox_px,
            "mineralInfo": {"name": cn, "formula": "", "hardness": "", "luster": "", "color": "", "description": desc[:150]}
        }], ensure_ascii=False, indent=2)

        prompt = random.choice(DETECTION_PROMPTS)
        img_texts.append(format_chatml(DETECTION_SYSTEM, prompt, response))
        img_images.append(item['image'])

vl_ds = Dataset.from_dict({"images": img_images, "text": img_texts}).cast_column("images", Image())
vl_splits = vl_ds.train_test_split(test_size=0.1, seed=42)
print(f"  Image detection: train={len(vl_splits['train'])}, eval={len(vl_splits['test'])}")

# ============================================================
# 加载模型
# ============================================================
print("\n" + "=" * 50)
print("  加载模型")
print("=" * 50)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.float16,
    load_in_4bit=True,
)

model = FastVisionModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ============================================================
# 训练 1: 文本问答
# ============================================================
print("\n" + "=" * 50)
print("  [1/2] 训练文本问答 (96 texts)")
print("=" * 50)

text_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "text_checkpoint"),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=80,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=40,
    eval_strategy="steps",
    eval_steps=40,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42, report_to="none",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    remove_unused_columns=True,
)

trainer1 = SFTTrainer(
    model=model, tokenizer=tokenizer, args=text_args,
    train_dataset=text_ds["train"],
    eval_dataset=text_ds["eval"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
)
# trainer1.train()

# ============================================================
# 训练 2: 图片检测
# ============================================================
print("\n" + "=" * 50)
print("  [2/2] 训练图片检测 (270 images)")
print("=" * 50)

# 用 processor 逐条预处理，让 dataset 自带 input_ids/pixel_values
print("Preprocessing VL dataset with processor...")
# 只加载一次 processor (不要每条都加载，太慢)
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained(MODEL_NAME)

def preprocess_vl(example):
    result = proc(
        text=example["text"],
        images=example["images"],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    return {
        "input_ids": result["input_ids"][0],
        "attention_mask": result["attention_mask"][0],
        "pixel_values": result["pixel_values"][0],
        "labels": result["input_ids"][0].clone(),
    }

# 预处理训练集和验证集
vl_train = vl_splits["train"].map(preprocess_vl, remove_columns=["text", "images"])
vl_eval = vl_splits["test"].map(preprocess_vl, remove_columns=["text", "images"])

# 设置格式为 torch tensors
vl_train.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values", "labels"])
vl_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values", "labels"])

print(f"  Train: {len(vl_train)}, Eval: {len(vl_eval)}")
print(f"  Sample keys: {list(vl_train[0].keys())}")

img_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "img_checkpoint"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=100,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42, report_to="none",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    remove_unused_columns=True,
)

trainer2 = SFTTrainer(
    model=model, tokenizer=tokenizer, args=img_args,
    train_dataset=vl_train,
    eval_dataset=vl_eval,
    max_seq_length=MAX_SEQ_LENGTH,
)
trainer2.train()

# ============================================================
# 保存
# ============================================================
print("\n" + "=" * 50)
print("  保存模型")
print("=" * 50)

model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
model.save_pretrained_merged(os.path.join(OUTPUT_DIR, "merged_16bit"), tokenizer, save_method="merged_16bit")
model.save_pretrained_gguf(os.path.join(OUTPUT_DIR, "gguf"), tokenizer, quantization_method="q4_k_m")
print(f"\nDone! Saved to: {OUTPUT_DIR}")
