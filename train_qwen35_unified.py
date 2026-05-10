"""
Qwen3.5-2B 统一训练 — 文本问答 + 矿物识别
Step 1: 训练文本问答 (纯文本 ChatML)
Step 2: 训练矿物识别 (图片 → 矿物信息 JSON)
QLoRA, 适配 6GB VRAM
"""

import torch
from unsloth import FastVisionModel
from datasets import load_dataset, Dataset
from PIL import Image
from transformers import TrainingArguments, AutoProcessor
from trl import SFTTrainer
import os, json, random

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"

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
    "你是一个专业的矿物识别专家。请仔细分析图片中的矿物，"
    "以JSON格式输出识别结果，包含以下字段："
    "label(矿物名称)、confidence(置信度0-1)、formula(化学式)、"
    "hardness(硬度)、luster(光泽)、color(颜色)、origin(产地)、"
    "uses(用途)、description(描述)。"
    "如果无法识别，label输出\"未知\"。"
)

MINERAL_INFO = {
    "石英": {
        "name": "石英", "formula": "SiO2", "hardness": "7",
        "luster": "玻璃光泽", "color": "无色/白色/紫色/粉色/烟色等",
        "origin": "全球广泛分布，伟晶岩、热液脉",
        "uses": "光学玻璃、电子元件、石英钟表、宝石饰品",
        "description": "石英是地壳中含量第二多的矿物，化学性质稳定，常见六方柱状晶体。品种包括水晶、紫水晶、蔷薇石英、烟晶、玛瑙等。"
    },
    "紫水晶": {
        "name": "紫水晶", "formula": "SiO2(含Fe)", "hardness": "7",
        "luster": "玻璃光泽", "color": "紫色",
        "origin": "巴西、乌拉圭、赞比亚、中国",
        "uses": "宝石饰品、标本收藏",
        "description": "紫水晶是石英的紫色变种，因铁离子和天然辐射形成色心而呈紫色。加热至400-500°C会褪色变为黄水晶。"
    },
    "黄铁矿": {
        "name": "黄铁矿", "formula": "FeS2", "hardness": "6-6.5",
        "luster": "金属光泽", "color": "金黄色",
        "origin": "热液矿床、沉积岩、煤层",
        "uses": "提取硫磺、制造硫酸、矿物标本",
        "description": "黄铁矿是最常见的硫化物矿物，常形成完美立方体晶体，因金黄色金属光泽常被误认为黄金，俗称愚人金。"
    },
    "方解石": {
        "name": "方解石", "formula": "CaCO3", "hardness": "3",
        "luster": "玻璃光泽", "color": "无色/白色/浅黄色",
        "origin": "沉积岩、变质岩、热液脉",
        "uses": "建筑材料、水泥制造、光学仪器、冶金熔剂",
        "description": "方解石是分布最广的碳酸盐矿物，具有菱面体解理和双折射现象，遇酸会冒泡。是石灰岩和大理岩的主要成分。"
    },
    "萤石": {
        "name": "萤石", "formula": "CaF2", "hardness": "4",
        "luster": "玻璃光泽", "color": "紫色/绿色/蓝色/黄色/无色",
        "origin": "热液矿床，中国浙江武义为著名产地",
        "uses": "冶金助熔剂、氢氟酸制造、光学镜片、宝石饰品",
        "description": "萤石颜色极其丰富，紫外线下可发荧光。fluorescence（荧光）一词即源于萤石。八面体解理，硬度较低需防刮。"
    },
    "赤铁矿": {
        "name": "赤铁矿", "formula": "Fe2O3", "hardness": "5.5-6.5",
        "luster": "金属至半金属光泽", "color": "钢灰色至黑色",
        "origin": "沉积变质矿床、热液矿床",
        "uses": "最重要的铁矿石，用于炼铁炼钢",
        "description": "赤铁矿是最重要的铁矿石矿物，条痕为特征性的樱红色。外表可呈钢灰色、红褐色等，但条痕始终是樱红色。"
    },
    "磁铁矿": {
        "name": "磁铁矿", "formula": "Fe3O4", "hardness": "5.5-6.5",
        "luster": "金属至半金属光泽", "color": "铁黑色",
        "origin": "岩浆岩、变质岩、砂矿",
        "uses": "铁矿石、磁性材料、指南针原料",
        "description": "磁铁矿是自然界磁性最强的矿物，铁黑色，条痕黑色。古代指南针即用天然磁铁矿制成。"
    },
    "孔雀石": {
        "name": "孔雀石", "formula": "Cu2CO3(OH)2", "hardness": "3.5-4",
        "luster": "玻璃至丝绢光泽", "color": "翠绿色",
        "origin": "铜矿床氧化带，非洲刚果、俄罗斯乌拉尔",
        "uses": "饰品、矿物颜料、标本收藏",
        "description": "孔雀石是铜的碳酸盐矿物，翠绿色，常有同心条带状花纹。常与蓝铜矿共生，古时用作绿色矿物颜料（石绿）。"
    },
    "蓝铜矿": {
        "name": "蓝铜矿", "formula": "Cu3(CO3)2(OH)2", "hardness": "3.5-4",
        "luster": "玻璃光泽", "color": "深蓝色",
        "origin": "铜矿床氧化带",
        "uses": "矿物颜料（石青）、标本收藏",
        "description": "蓝铜矿是铜的碳酸盐矿物，呈鲜艳深蓝色。常与孔雀石共生，在空气中会缓慢转变为孔雀石。"
    },
    "石膏": {
        "name": "石膏", "formula": "CaSO4·2H2O", "hardness": "2",
        "luster": "玻璃至珍珠光泽", "color": "无色/白色/灰色",
        "origin": "蒸发岩沉积，盐湖和潟湖环境",
        "uses": "建筑材料、医疗石膏、模具制作、农业改良",
        "description": "石膏是最软的矿物之一，可用指甲刮动。加热失去部分结晶水后变成熟石膏，加水可重新硬化。"
    },
    "方铅矿": {
        "name": "方铅矿", "formula": "PbS", "hardness": "2.5",
        "luster": "金属光泽", "color": "铅灰色",
        "origin": "热液矿床、铅锌矿床",
        "uses": "提炼铅，用于铅酸电池、防辐射材料",
        "description": "方铅矿是最重要的铅矿石，具有完美的立方体解理，比重极大(约7.5)，手感沉重。"
    },
    "辰砂": {
        "name": "辰砂", "formula": "HgS", "hardness": "2-2.5",
        "luster": "金刚至金属光泽", "color": "鲜红色",
        "origin": "低温热液矿床，中国贵州万山为著名产地",
        "uses": "汞矿石、红色颜料、中药(慎用)",
        "description": "辰砂又称朱砂，是最主要的汞矿石。颜色鲜红，条痕红色。含汞有毒性，加热会释放汞蒸气。"
    },
    "自然金": {
        "name": "自然金", "formula": "Au", "hardness": "2.5-3",
        "luster": "金属光泽", "color": "金黄色",
        "origin": "热液矿床、砂矿",
        "uses": "贵金属、珠宝首饰、电子元件、货币储备",
        "description": "自然金是天然产出的金元素矿物，具极好的延展性，化学性质稳定不氧化。常含少量银形成天然合金。"
    },
    "自然铜": {
        "name": "自然铜", "formula": "Cu", "hardness": "2.5-3",
        "luster": "金属光泽", "color": "铜红色",
        "origin": "铜矿床氧化带、玄武岩气孔",
        "uses": "铜金属来源、电线电缆、电子工业",
        "description": "自然铜是天然产出的铜元素矿物，具延展性。暴露空气中表面会氧化变暗，失去金属光泽。"
    },
    "硫磺": {
        "name": "硫磺", "formula": "S", "hardness": "1.5-2.5",
        "luster": "油脂至金刚光泽", "color": "柠檬黄色",
        "origin": "火山喷气孔、沉积蒸发岩",
        "uses": "制造硫酸、化肥、火药、橡胶硫化",
        "description": "自然硫呈特征的柠檬黄色，极易燃烧产生蓝色火焰和刺激性二氧化硫气体。硬度很低。"
    },
    "绿柱石": {
        "name": "绿柱石", "formula": "Be3Al2Si6O18", "hardness": "7.5-8",
        "luster": "玻璃光泽", "color": "绿色/蓝色/粉色/黄色/无色",
        "origin": "伟晶岩、热液脉",
        "uses": "铍矿石、宝石（祖母绿、海蓝宝石）",
        "description": "绿柱石是铍的主要矿石矿物。宝石变种包括祖母绿（绿色）、海蓝宝石（蓝色）和摩根石（粉色）。"
    },
    "刚玉": {
        "name": "刚玉", "formula": "Al2O3", "hardness": "9",
        "luster": "玻璃至金刚光泽", "color": "红色/蓝色/黄色/粉色/无色等",
        "origin": "变质岩、岩浆岩",
        "uses": "宝石饰品（红宝石和蓝宝石）、研磨材料、钟表轴承",
        "description": "刚玉硬度仅次于金刚石。红色变种称红宝石，其他颜色统称蓝宝石。广泛用于精密仪器轴承和高级研磨材料。"
    },
    "黄玉": {
        "name": "黄玉", "formula": "Al2SiO4(F,OH)2", "hardness": "8",
        "luster": "玻璃光泽", "color": "无色/黄色/蓝色/粉色",
        "origin": "伟晶岩、热液脉",
        "uses": "宝石饰品（托帕石）、标本收藏",
        "description": "黄玉硬度8，斜方晶系柱状晶体，具有一组完全解理。宝石级黄玉又称托帕石，蓝色托帕石多为辐照处理。"
    },
    "电气石": {
        "name": "电气石", "formula": "含硼环状硅酸盐(复杂)", "hardness": "7-7.5",
        "luster": "玻璃光泽", "color": "黑色/绿色/红色/蓝色/多色",
        "origin": "伟晶岩、热液脉",
        "uses": "宝石饰品（碧玺）、压电元件、空气净化",
        "description": "电气石又称碧玺，具有热电性和压电性，加热或加压时两端产生电荷。颜色极为丰富，帕拉伊巴碧玺最为珍贵。"
    },
    "闪锌矿": {
        "name": "闪锌矿", "formula": "ZnS", "hardness": "3.5-4",
        "luster": "油脂至金刚光泽", "color": "黄色/棕色/黑色",
        "origin": "热液矿床、铅锌矿床",
        "uses": "锌的最主要矿石，用于镀锌、黄铜、电池",
        "description": "闪锌矿是锌的最主要矿石矿物，含铁越高颜色越深。具有六组完全解理，常与方铅矿共生。"
    },
}

DETECTION_PROMPTS = [
    "请识别图片中的矿物。",
    "这是什么矿物？",
    "这张图片是什么矿物？",
]

def format_chatml(system, instruction, output):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

def format_detection_response(cn_name):
    info = MINERAL_INFO.get(cn_name)
    if not info:
        return json.dumps({"label": cn_name, "confidence": 0.7}, ensure_ascii=False)
    confidence = round(random.uniform(0.85, 0.99), 2)
    return json.dumps({
        "label": info["name"],
        "confidence": confidence,
        "formula": info["formula"],
        "hardness": info["hardness"],
        "luster": info["luster"],
        "color": info["color"],
        "origin": info["origin"],
        "uses": info["uses"],
        "description": info["description"],
    }, ensure_ascii=False)

def create_labels(input_ids, tokenizer):
    labels = input_ids.clone()
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_start_id is None or im_end_id is None or im_start_id == tokenizer.unk_token_id:
        return labels
    start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
    if len(start_positions) == 0:
        return labels
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
    if assistant_token_id is None or assistant_token_id == tokenizer.unk_token_id:
        return labels
    assistant_start = None
    for idx in range(len(start_positions) - 1, -1, -1):
        pos = start_positions[idx].item()
        if pos + 1 < len(input_ids) and input_ids[pos + 1].item() == assistant_token_id:
            assistant_start = pos
            break
    if assistant_start is None:
        return labels
    newline_id = tokenizer.convert_tokens_to_ids("\n")
    content_start = assistant_start + 2
    if content_start < len(input_ids) and input_ids[content_start].item() == newline_id:
        content_start += 1
    content_end = len(input_ids)
    for i in range(content_start, len(input_ids)):
        if input_ids[i].item() == im_end_id:
            content_end = i
            break
    labels[:content_start] = -100
    labels[content_end:] = -100
    return labels

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
# Step 2: 矿物识别数据 (图片 → 矿物信息JSON)
# ============================================================
print("\n" + "=" * 50)
print("  Step 2: 准备矿物识别数据")
print("=" * 50)

IMG_DIR = r"D:\OneDrive\Desktop\train\img"

img_images, img_texts = [], []
file_count = 0
for folder in os.listdir(IMG_DIR):
    folder_path = os.path.join(IMG_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    cn_name = folder.split("_")[0]
    if cn_name not in MINERAL_INFO:
        continue

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            continue
        try:
            img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
            response = format_detection_response(cn_name)
            prompt = random.choice(DETECTION_PROMPTS)
            img_texts.append(format_chatml(DETECTION_SYSTEM, prompt, response))
            img_images.append(img)
            file_count += 1
        except Exception as e:
            print(f"  WARNING: 跳过 {fname}: {e}")

print(f"  加载本地图片: {file_count} 张")

vl_ds = Dataset.from_dict({"images": img_images, "text": img_texts})
vl_splits = vl_ds.train_test_split(test_size=0.1, seed=42)

print(f"  Mineral ID: train={len(vl_splits['train'])}, eval={len(vl_splits['test'])}")

# ============================================================
# 加载模型
# ============================================================
print("\n" + "=" * 50)
print("  加载模型")
print("=" * 50)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.float16, load_in_4bit=True,
)

model = FastVisionModel.get_peft_model(
    model, r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)

# ============================================================
# 训练 1: 文本问答
# ============================================================
print("\n" + "=" * 50)
print("  [1/2] 训练文本问答")
print("=" * 50)

text_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "text_checkpoint"),
    per_device_train_batch_size=2, per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5, max_steps=80, learning_rate=2e-4,
    logging_steps=10, save_steps=40, eval_strategy="no",
    optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine",
    seed=42, report_to="none", dataloader_num_workers=0,
    remove_unused_columns=True,
)

trainer1 = SFTTrainer(
    model=model, tokenizer=tokenizer, args=text_args,
    train_dataset=text_ds["train"], eval_dataset=text_ds["eval"],
    dataset_text_field="text", max_seq_length=MAX_SEQ_LENGTH, packing=False,
)
# trainer1.train()

# ============================================================
# 训练 2: 矿物识别
# ============================================================
print("\n" + "=" * 50)
print(f"  [2/2] 训练矿物识别")
print("=" * 50)

print("Preprocessing with processor...")
proc = AutoProcessor.from_pretrained(MODEL_NAME)

def preprocess_vl(example):
    result = proc(text=example["text"], images=example["images"],
                  return_tensors="pt", padding=False, truncation=True, max_length=MAX_SEQ_LENGTH,
                  return_mm_token_type_ids=True)
    input_ids = result["input_ids"][0]
    labels = create_labels(input_ids, proc.tokenizer)
    return {
        "input_ids": input_ids,
        "attention_mask": result["attention_mask"][0],
        "pixel_values": result["pixel_values"][0],
        "image_grid_thw": result["image_grid_thw"][0],
        "mm_token_type_ids": result["mm_token_type_ids"][0],
        "labels": labels,
    }

vl_train = vl_splits["train"].map(preprocess_vl, remove_columns=["text", "images"])
vl_eval = vl_splits["test"].map(preprocess_vl, remove_columns=["text", "images"])
vl_train.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids", "labels"])
vl_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids", "labels"])

img_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "img_checkpoint"),
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10, max_steps=800, learning_rate=2e-4,
    logging_steps=50, save_steps=200, eval_strategy="no",
    optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine",
    seed=42, report_to="none", dataloader_num_workers=0,
    remove_unused_columns=True,
)

trainer2 = SFTTrainer(
    model=model, tokenizer=tokenizer, args=img_args,
    train_dataset=vl_train, eval_dataset=vl_eval,
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
