"""
================================================================================
Qwen3.5-2B 统一训练脚本 — 文本问答 + 矿物识别
================================================================================

训练流程分为两个阶段：
  Step 1: 训练文本问答  —— 输入：文字问题，输出：文字回答（纯文本 ChatML 格式）
  Step 2: 训练矿物识别  —— 输入：矿物图片，输出：矿物信息 JSON

技术要点：
  - 使用 QLoRA（量化低秩适配）微调，可在 6GB 显存的 GPU 上运行
  - 使用 unsloth 库加速训练
  - 使用 HuggingFace trl 库的 SFTTrainer 进行监督微调（SFT）

================================================================================
"""

# ==============================================================================
# 标准库 & 第三方库导入
# ==============================================================================

import os       # 操作系统接口，用于文件路径拼接、目录遍历等（类似 Java 的 java.io.File）
import json     # JSON 序列化/反序列化（类似 Java 的 Gson / Jackson）
import random   # 随机数生成（类似 Java 的 java.util.Random）

import torch    # PyTorch 深度学习框架，提供张量运算（类似 Java 的 NDArray）

from unsloth import FastVisionModel             # unsloth 对视觉-语言模型的高性能封装
from datasets import load_dataset, Dataset      # HuggingFace 数据集加载工具
from PIL import Image                           # Python Imaging Library，用于图片读取
from transformers import TrainingArguments, AutoProcessor   # HuggingFace 训练参数 & 多模态处理器
from trl import SFTTrainer                      # HuggingFace TRL 的监督微调训练器


# ==============================================================================
# 环境变量设置
# 注：Python 的 os.environ 相当于 Java 的 System.setProperty()
#     这里设置为"离线模式"，防止训练时联网下载模型（已提前下载到本地）
# ==============================================================================

os.environ["HF_HUB_OFFLINE"] = "1"             # 禁止 HuggingFace Hub 联网
os.environ["TRANSFORMERS_OFFLINE"] = "1"        # 禁止 Transformers 库联网
os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"  # 关闭 unsloth 融合 Loss，提高兼容性


# ==============================================================================
# 全局随机种子设置
# 目的：让每次运行的随机行为完全一致，保证实验可复现（Reproducibility）
# 42 只是一个惯例数字，可以换成任何整数
# ==============================================================================

random.seed(42)         # 设置 Python 内置随机模块的种子
torch.manual_seed(42)   # 设置 PyTorch 张量操作的随机种子


# ==============================================================================
# 全局配置常量
# 注：Python 惯例用全大写命名常量（类似 Java 的 static final）
# ==============================================================================

MODEL_NAME     = "Qwen/Qwen3.5-2B"   # 基础模型名称（本地离线路径或 HuggingFace 模型 ID）
MAX_SEQ_LENGTH = 1024                 # 最大序列长度（Token 数量），超出部分会被截断
LORA_R         = 16                   # LoRA 的秩（Rank），控制可训练参数量；越大效果越好但显存占用越多
LORA_ALPHA     = 16                   # LoRA 缩放系数，通常与 LORA_R 相同

# 训练输出目录（存放 checkpoint、最终模型）
OUTPUT_DIR = "D:\\OneDrive\\Desktop\\train\\mineral_unified_model"

# 训练数据目录（存放 train.json、eval.json）
DATA_DIR   = "D:\\OneDrive\\Desktop\\train\\data"


# ==============================================================================
# 系统提示词（System Prompt）
# 作用：在每次对话开头告诉模型它扮演什么角色、遵循什么规则
#       类似于给员工的"岗位职责说明书"
# ==============================================================================

# 用于"文本问答"阶段的系统提示词
SYSTEM_PROMPT = (
    "你是一个专业的矿物识别助手，专门帮助用户了解矿物相关知识。"
    "请用简洁、准确、友好的中文回答用户的问题。\n\n"
    "如果用户的问题超出你的知识范围，请诚实地告知，不要编造信息。"
)

# 用于"矿物图片识别"阶段的系统提示词
# 明确要求模型输出 JSON 格式，并规定了必须包含的字段
DETECTION_SYSTEM = (
    "你是一个专业的矿物识别专家。请仔细分析图片中的矿物，"
    "以JSON格式输出识别结果，包含以下字段："
    "label(矿物名称)、confidence(置信度0-1)、formula(化学式)、"
    "hardness(硬度)、luster(光泽)、color(颜色)、origin(产地)、"
    "uses(用途)、description(描述)。"
    "如果无法识别，label输出\"未知\"。"
)


# ==============================================================================
# 矿物知识库
# 数据结构：dict（字典），相当于 Java 的 HashMap<String, Map<String, String>>
# Key   = 矿物中文名称
# Value = 该矿物的详细属性字典
# 这份知识库用于：
#   1. 判断图片文件夹名称是否为已知矿物（用于过滤无效数据）
#   2. 生成矿物识别训练样本的标准答案（Ground Truth）
# ==============================================================================

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

# 矿物识别任务的用户提问模板列表
# 训练时会从中随机选一句，增加数据多样性，防止模型过拟合特定表达
DETECTION_PROMPTS = [
    "请识别图片中的矿物。",
    "这是什么矿物？",
    "这张图片是什么矿物？",
]


# ==============================================================================
# 工具函数区
# ==============================================================================

def format_chatml(system: str, instruction: str, output: str) -> str:
    """
    将三部分内容拼装为 ChatML 格式的训练文本。

    ChatML 是一种对话标记格式，告诉模型：
      - 哪段文字是"系统指令"（system）
      - 哪段文字是"用户问题"（user）
      - 哪段文字是"模型回答"（assistant）

    格式示例：
      <|im_start|>system
      你是...
      <|im_end|>
      <|im_start|>user
      这是什么矿物？
      <|im_end|>
      <|im_start|>assistant
      {"label": "石英", ...}
      <|im_end|>

    参数：
        system      (str): 系统提示词，定义模型角色
        instruction (str): 用户提问内容
        output      (str): 期望的模型回答（训练目标）

    返回：
        str: 拼装好的 ChatML 格式字符串
    """
    # Python f-string：用花括号嵌入变量，类似 Java 的 String.format()
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def format_detection_response(cn_name: str) -> str:
    """
    根据矿物中文名称，生成标准的 JSON 格式识别结果（作为训练标签）。

    逻辑：
      1. 在 MINERAL_INFO 字典中查找该矿物的详细信息
      2. 如果找不到，返回一个仅含名称和默认置信度的简化 JSON
      3. 如果找到，生成一个随机置信度（0.85 ~ 0.99），并填入所有字段

    注意：置信度随机化是为了让训练数据更真实，
         避免模型学到"所有矿物置信度都一样"这种错误规律。

    参数：
        cn_name (str): 矿物中文名称，例如 "石英"

    返回：
        str: JSON 格式的识别结果字符串（ensure_ascii=False 保证中文不被转义）
    """
    info = MINERAL_INFO.get(cn_name)  # dict.get() 找不到时返回 None，不会抛异常

    # 未知矿物：返回简化 JSON
    if not info:
        return json.dumps({"label": cn_name, "confidence": 0.7}, ensure_ascii=False)

    # 生成随机置信度，round() 保留 2 位小数
    confidence = round(random.uniform(0.85, 0.99), 2)

    # 构建完整的识别结果字典，并序列化为 JSON 字符串
    return json.dumps({
        "label":       info["name"],
        "confidence":  confidence,
        "formula":     info["formula"],
        "hardness":    info["hardness"],
        "luster":      info["luster"],
        "color":       info["color"],
        "origin":      info["origin"],
        "uses":        info["uses"],
        "description": info["description"],
    }, ensure_ascii=False)


def create_labels(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    根据 input_ids（Token 序列），生成训练用的 labels 张量。

    【背景知识】
    语言模型的训练方式是"下一个 Token 预测"：给定前 N 个 Token，预测第 N+1 个。
    但我们不希望模型学习如何生成"系统提示"或"用户问题"部分——
    那些是"输入"，我们只希望模型学习如何生成"助手回答"部分。

    实现方式：
      - 将 labels 初始化为 input_ids 的副本（形状完全相同的张量）
      - 把"非助手回答"部分的 label 全部设为 -100
      - PyTorch 的 CrossEntropyLoss 会自动忽略值为 -100 的位置

    具体步骤：
      1. 找到序列中所有 <|im_start|> 标记的位置
      2. 从后往前遍历，找到最后一个 <|im_start|>assistant 的位置
      3. 确定助手回答的起始和结束位置（跳过 "assistant\n"）
      4. 将助手回答之前的所有 label 设为 -100
      5. 将助手回答的 <|im_end|> 之后的所有 label 设为 -100

    参数：
        input_ids (torch.Tensor): 形如 [seq_len] 的 1D 整数张量，代表 Token ID 序列
        tokenizer              : 用于将特殊字符串转换为 Token ID

    返回：
        torch.Tensor: 与 input_ids 形状相同的 labels 张量，非目标位置值为 -100
    """
    # clone() 深拷贝张量，避免修改原始 input_ids（类似 Java 的 clone()）
    labels = input_ids.clone()

    # 获取特殊 Token 的 ID
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # 如果特殊 Token 不存在（返回 None 或 unk_token_id），直接返回原始 labels
    # unk_token_id 是"未知 Token"，说明词表中找不到该 Token
    if im_start_id is None or im_end_id is None or im_start_id == tokenizer.unk_token_id:
        return labels

    # nonzero() 找到张量中所有等于 im_start_id 的位置索引
    # as_tuple=True 返回元组形式；[0] 取第一个维度（1D 张量只有一个维度）
    start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]

    # 如果序列中没有任何 <|im_start|>，直接返回
    if len(start_positions) == 0:
        return labels

    # 获取 "assistant" 字符串对应的 Token ID
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
    if assistant_token_id is None or assistant_token_id == tokenizer.unk_token_id:
        return labels

    # 从后往前遍历所有 <|im_start|> 位置，找到最后一个 <|im_start|>assistant
    # 即找到"最后一轮对话"中助手回答的起始标记
    # range(len-1, -1, -1) 相当于 Java 的 for(int i=len-1; i>=0; i--)
    assistant_start = None
    for idx in range(len(start_positions) - 1, -1, -1):
        pos = start_positions[idx].item()  # .item() 将张量标量转为 Python 原生 int
        # 检查 <|im_start|> 的下一个 Token 是否为 "assistant"
        if pos + 1 < len(input_ids) and input_ids[pos + 1].item() == assistant_token_id:
            assistant_start = pos
            break

    # 如果没有找到 assistant 段，直接返回
    if assistant_start is None:
        return labels

    # 跳过 "<|im_start|>assistant\n"，定位到助手回答内容的真正起始位置
    # assistant_start     → <|im_start|>
    # assistant_start + 1 → assistant
    # assistant_start + 2 → \n（换行符，需要额外跳过）
    newline_id    = tokenizer.convert_tokens_to_ids("\n")
    content_start = assistant_start + 2
    if content_start < len(input_ids) and input_ids[content_start].item() == newline_id:
        content_start += 1  # 跳过换行符，到达真正的回答内容

    # 从 content_start 往后找第一个 <|im_end|>，确定助手回答的结束位置
    content_end = len(input_ids)  # 默认到序列末尾
    for i in range(content_start, len(input_ids)):
        if input_ids[i].item() == im_end_id:
            content_end = i
            break

    # 将助手回答内容之外的所有位置标记为 -100（训练时忽略这些位置的 loss）
    labels[:content_start] = -100   # 屏蔽 system + user 部分
    labels[content_end:]   = -100   # 屏蔽结束符之后的部分（padding 等）

    return labels


# ==============================================================================
# 数据集构建函数
# ==============================================================================

def build_text_dataset(examples: dict) -> dict:
    """
    将文本问答数据批量转换为 ChatML 格式。

    【背景知识】
    HuggingFace datasets 的 .map() 方法支持 batched=True，
    此时传入的 examples 是一个 dict，每个 key 对应一个列表（而非单条数据）。
    这类似于 Java 中将 List<Map> 转置为 Map<List> 的操作。

    例如，输入 examples 结构：
      {
        "system": ["提示词A", "提示词B", ...],
        "instruction": ["问题A", "问题B", ...],
        "output": ["答案A", "答案B", ...]
      }

    参数：
        examples (dict): 批量数据字典，包含 "system"、"instruction"、"output" 三列

    返回：
        dict: 包含 "text" 键的字典，值为 ChatML 格式字符串列表
    """
    texts = []
    # 使用 range(len(...)) 遍历批次中每一条数据的索引
    for i in range(len(examples["instruction"])):
        texts.append(format_chatml(
            examples["system"][i],
            examples["instruction"][i],
            examples["output"][i],
        ))
    return {"text": texts}


def load_text_qa_dataset(data_dir: str):
    """
    从磁盘加载文本问答数据集（JSON 格式），并转换为 ChatML 训练格式。

    数据文件结构（train.json / eval.json）：
      每行一个 JSON 对象，包含 system、instruction、input、output 字段。

    参数：
        data_dir (str): 数据目录路径

    返回：
        DatasetDict: 包含 "train" 和 "eval" 两个子集的 HuggingFace 数据集对象
    """
    print("=" * 50)
    print("  Step 1: 准备文本问答数据")
    print("=" * 50)

    # load_dataset 支持从本地 JSON 文件加载，data_files 指定训练集和验证集路径
    # 类似 Java 中读取 JSON 文件并反序列化为对象列表
    text_ds = load_dataset("json", data_files={
        "train": os.path.join(data_dir, "train.json"),
        "eval":  os.path.join(data_dir, "eval.json"),
    })

    # .map() 对数据集中每条（或每批）数据执行转换函数
    # batched=True：批量处理（更高效）
    # remove_columns：删除不再需要的原始列（只保留新增的 "text" 列）
    text_ds = text_ds.map(
        build_text_dataset,
        batched=True,
        remove_columns=["system", "instruction", "input", "output"],
    )

    print(f"  Text QA: train={len(text_ds['train'])}, eval={len(text_ds['eval'])}")
    return text_ds


def load_mineral_image_dataset(img_dir: str):
    """
    从本地图片目录加载矿物识别数据集。

    目录结构约定：
      img_dir/
        石英_001/         <- 文件夹名以矿物中文名开头，用下划线分隔编号
          img_001.jpg
          img_002.jpg
        黄铁矿_002/
          img_001.jpg
          ...

    处理逻辑：
      1. 遍历 img_dir 下的所有子文件夹
      2. 从文件夹名中提取矿物名（取下划线前的部分）
      3. 检查该矿物名是否在 MINERAL_INFO 知识库中
      4. 遍历文件夹内所有图片文件（支持 jpg/jpeg/png/bmp/webp）
      5. 用 PIL 打开图片，转换为 RGB（统一格式，去掉 RGBA 的透明通道）
      6. 生成对应的 ChatML 格式训练文本
      7. 构建 Dataset 并按 9:1 比例划分训练集和验证集

    参数：
        img_dir (str): 图片根目录路径

    返回：
        DatasetDict: 包含 "train" 和 "test" 两个子集的数据集对象
    """
    print("\n" + "=" * 50)
    print("  Step 2: 准备矿物识别数据")
    print("=" * 50)

    # 支持的图片文件扩展名集合（小写）
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    img_images = []  # 存储 PIL Image 对象列表（Python list，类似 Java 的 ArrayList<BufferedImage>）
    img_texts  = []  # 存储对应的 ChatML 格式文本列表
    file_count = 0   # 成功加载的图片计数

    # os.listdir() 列出目录下所有文件/子目录名（不保证顺序）
    for folder in os.listdir(img_dir):
        folder_path = os.path.join(img_dir, folder)

        # os.path.isdir() 判断是否为目录（跳过普通文件）
        if not os.path.isdir(folder_path):
            continue

        # 从文件夹名称中提取矿物中文名
        # "石英_001".split("_") → ["石英", "001"]；[0] 取第一个元素
        cn_name = folder.split("_")[0]

        # 检查该矿物名是否在知识库中（不在则跳过整个文件夹）
        if cn_name not in MINERAL_INFO:
            continue

        # 对文件夹内的文件名进行排序，保证处理顺序一致（可复现）
        for fname in sorted(os.listdir(folder_path)):
            # str.lower() 转小写；str.endswith() 检查后缀（支持元组参数）
            if not fname.lower().endswith(SUPPORTED_EXTENSIONS):
                continue

            try:
                # 打开图片并转为 RGB（统一色彩空间，去掉透明通道）
                img = Image.open(os.path.join(folder_path, fname)).convert("RGB")

                # 生成该图片对应的标准 JSON 答案
                response = format_detection_response(cn_name)

                # 随机选择一个用户提问模板（增加数据多样性）
                prompt = random.choice(DETECTION_PROMPTS)

                # 将文本组装为 ChatML 格式
                img_texts.append(format_chatml(DETECTION_SYSTEM, prompt, response))
                img_images.append(img)
                file_count += 1

            except Exception as e:
                # try/except 类似 Java 的 try/catch；Exception 是所有异常的基类
                print(f"  WARNING: 跳过 {fname}: {e}")

    print(f"  加载本地图片: {file_count} 张")

    # Dataset.from_dict() 从字典构建 HuggingFace Dataset 对象
    # 类似 Java 中将多个 List 合并为一个 List<Map>
    vl_ds = Dataset.from_dict({"images": img_images, "text": img_texts})

    # train_test_split() 随机划分数据集，test_size=0.1 表示 10% 作为验证集
    # seed=42 保证每次划分结果一致
    vl_splits = vl_ds.train_test_split(test_size=0.1, seed=42)

    print(f"  Mineral ID: train={len(vl_splits['train'])}, eval={len(vl_splits['test'])}")
    return vl_splits


def load_model_and_apply_lora(model_name: str, max_seq_length: int, lora_r: int, lora_alpha: int):
    """
    加载预训练模型并附加 LoRA 适配器（低秩适配微调）。

    【背景知识：QLoRA 微调】
    - 直接微调大模型需要修改所有权重，显存消耗极大
    - LoRA 的核心思想：冻结原始权重，只在关键层旁边插入"低秩矩阵"进行训练
      相当于：原权重矩阵 W（大） ≈ W + A × B（A、B 是小矩阵，参数量少得多）
    - QLoRA 在此基础上用 4-bit 量化加载原始权重，进一步压缩显存

    target_modules 说明（这些是 Transformer 注意力层的核心投影矩阵）：
      q_proj  → Query 投影
      k_proj  → Key 投影
      v_proj  → Value 投影
      o_proj  → 输出投影
      gate_proj / up_proj / down_proj → MLP（前馈网络）的投影层

    参数：
        model_name     (str): 模型路径或 HuggingFace 模型 ID
        max_seq_length (int): 最大序列长度
        lora_r         (int): LoRA 秩（Rank）
        lora_alpha     (int): LoRA 缩放系数

    返回：
        tuple: (model, tokenizer) —— 附加了 LoRA 的模型和对应的分词器
    """
    print("\n" + "=" * 50)
    print("  加载模型")
    print("=" * 50)

    # 以 4-bit 量化加载模型（大幅减少显存占用）
    # dtype=torch.float16：激活值和 LoRA 权重用 16-bit 浮点（精度与效率的平衡）
    # load_in_4bit=True：模型主体权重用 4-bit 量化
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    # 在模型上附加 LoRA 适配器
    # lora_dropout=0.0：不使用 Dropout（unsloth 推荐关闭以提速）
    # bias="none"：不训练偏置项
    # use_gradient_checkpointing="unsloth"：使用梯度检查点技术，用计算换显存
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
            "gate_proj", "up_proj", "down_proj",       # MLP 层
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model, tokenizer


def preprocess_vision_language_sample(example: dict, processor, tokenizer, max_seq_length: int) -> dict:
    """
    对单条视觉-语言样本进行预处理，将图片和文本编码为模型可接受的张量格式。

    处理流程：
      1. 使用 AutoProcessor 同时处理文本（Tokenize）和图片（Resize + Normalize）
      2. 提取 input_ids、attention_mask、pixel_values 等张量
      3. 调用 create_labels() 生成只监督助手回答部分的 labels

    【张量维度说明】
      result["input_ids"]   形状：[1, seq_len]（批次维度=1），取 [0] 得到 [seq_len]
      result["pixel_values"] 形状：[1, C, H, W]，取 [0] 得到 [C, H, W]

    参数：
        example        (dict): 包含 "text"（str）和 "images"（PIL.Image）的单条样本
        processor            : AutoProcessor 实例（同时处理文本和图片）
        tokenizer            : 分词器（用于 create_labels 内部的 Token ID 查询）
        max_seq_length (int) : 最大序列长度

    返回：
        dict: 包含模型训练所需的所有张量字段
    """
    # 调用 processor 同时编码文本和图片
    # return_tensors="pt"：返回 PyTorch 张量（pt = PyTorch，tf = TensorFlow）
    # padding=False：不补齐（后续 DataCollator 会统一处理）
    # truncation=True：超长时截断
    result = processor(
        text=example["text"],
        images=example["images"],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        return_mm_token_type_ids=True,  # 返回多模态 Token 类型标识（区分图片 Token 和文本 Token）
    )

    # 取 [0] 去掉 batch 维度（processor 输出默认包含 batch 维度）
    input_ids = result["input_ids"][0]

    # 生成训练 labels（仅监督助手回答部分）
    labels = create_labels(input_ids, tokenizer)

    return {
        "input_ids":         input_ids,
        "attention_mask":    result["attention_mask"][0],
        "pixel_values":      result["pixel_values"][0],
        "image_grid_thw":    result["image_grid_thw"][0],     # 图片的时间/高/宽网格信息（视觉编码用）
        "mm_token_type_ids": result["mm_token_type_ids"][0],  # 多模态 Token 类型 ID
        "labels":            labels,
    }


def prepare_vision_language_dataset(vl_splits, model_name: str, max_seq_length: int):
    """
    对矿物识别数据集进行批量预处理，转换为张量格式并设置为 PyTorch 格式。

    参数：
        vl_splits           : 包含 "train" 和 "test" 的原始数据集（含 PIL 图片）
        model_name     (str): 模型路径，用于加载 AutoProcessor
        max_seq_length (int): 最大序列长度

    返回：
        tuple: (vl_train, vl_eval) —— 预处理后的训练集和验证集
    """
    print("Preprocessing with processor...")

    # AutoProcessor 是 HuggingFace 提供的"自动处理器"
    # 对于多模态模型，它内部包含：文本 Tokenizer + 图片 Feature Extractor
    proc = AutoProcessor.from_pretrained(model_name)

    # 定义预处理函数（使用闭包捕获 proc 和 max_seq_length）
    # Python 中函数可以捕获外部变量，无需显式传参（类似 Java 的匿名内部类捕获 final 变量）
    def preprocess_fn(example):
        return preprocess_vision_language_sample(example, proc, proc.tokenizer, max_seq_length)

    # 对训练集和验证集分别应用预处理
    # remove_columns：处理完成后删除原始的 PIL 图片和文本列（节省内存）
    vl_train = vl_splits["train"].map(preprocess_fn, remove_columns=["text", "images"])
    vl_eval  = vl_splits["test"].map(preprocess_fn,  remove_columns=["text", "images"])

    # 将数据集格式设置为 PyTorch 张量，只保留训练所需的列
    tensor_columns = [
        "input_ids", "attention_mask", "pixel_values",
        "image_grid_thw", "mm_token_type_ids", "labels"
    ]
    vl_train.set_format(type="torch", columns=tensor_columns)
    vl_eval.set_format(type="torch",  columns=tensor_columns)

    return vl_train, vl_eval


def train_text_qa(model, tokenizer, text_ds, output_dir: str, max_seq_length: int):
    """
    阶段一：训练文本问答能力。

    使用 SFTTrainer（监督微调训练器）进行纯文本 ChatML 格式的训练。
    此阶段主要让模型学会用中文流畅地回答矿物知识问题。

    【训练超参数说明】
      per_device_train_batch_size=2  → 每个 GPU 每步处理 2 条样本
      gradient_accumulation_steps=4 → 累积 4 步梯度再更新（等效 batch_size=8）
      warmup_steps=5                → 前 5 步学习率从 0 线性增长到设定值（预热）
      max_steps=80                  → 总训练步数（不用 epoch，直接限制步数）
      learning_rate=2e-4            → 学习率 0.0002
      optim="adamw_8bit"            → 8-bit AdamW 优化器（节省显存）
      lr_scheduler_type="cosine"    → 余弦退火学习率调度器
      eval_strategy="no"            → 训练过程中不进行验证（节省时间）

    注意：
        当前 trainer1.train() 被注释掉，如需启用文本问答训练，请取消注释。

    参数：
        model          : 附加了 LoRA 的模型
        tokenizer      : 分词器
        text_ds        : 文本问答数据集（包含 "train" 和 "eval"）
        output_dir (str): 训练输出根目录
        max_seq_length (int): 最大序列长度
    """
    print("\n" + "=" * 50)
    print("  [1/2] 训练文本问答")
    print("=" * 50)

    text_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "text_checkpoint"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=80,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=40,
        eval_strategy="no",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",          # 不使用 Weights & Biases 等实验追踪工具
        dataloader_num_workers=0,  # Windows 上多进程数据加载可能有问题，设为 0 用主进程
        remove_unused_columns=True,
    )

    trainer1 = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=text_args,
        train_dataset=text_ds["train"],
        eval_dataset=text_ds["eval"],
        dataset_text_field="text",  # 指定数据集中存放 ChatML 文本的列名
        max_seq_length=max_seq_length,
        packing=False,              # 不将多条短样本打包成一条（保持每条完整）
    )

    # 注意：此处 trainer1.train() 被注释掉
    # 如果需要执行 Step 1 的文本问答训练，请取消下面这行的注释
    # trainer1.train()


def train_mineral_recognition(model, tokenizer, vl_train, vl_eval, output_dir: str, max_seq_length: int):
    """
    阶段二：训练矿物图片识别能力。

    使用已在文本问答上微调过的模型，继续在矿物图片数据集上训练，
    使模型能够根据图片输出结构化的 JSON 识别结果。

    【与文本训练的主要区别】
      - 数据集已预处理为张量（不使用 dataset_text_field）
      - batch_size=1（图片数据显存占用更大）
      - max_steps=800（图片任务更复杂，需要更多步数）

    参数：
        model          : 模型（已经过文本训练，或直接使用初始化后的 LoRA 模型）
        tokenizer      : 分词器
        vl_train       : 预处理后的矿物识别训练集（张量格式）
        vl_eval        : 预处理后的矿物识别验证集（张量格式）
        output_dir (str): 训练输出根目录
        max_seq_length (int): 最大序列长度
    """
    print("\n" + "=" * 50)
    print("  [2/2] 训练矿物识别")
    print("=" * 50)

    img_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "img_checkpoint"),
        per_device_train_batch_size=1,  # 图片数据显存消耗大，batch_size 设为 1
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # 等效 batch_size=4
        warmup_steps=10,
        max_steps=800,                  # 图片任务训练步数更多
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=200,
        eval_strategy="no",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer2 = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=img_args,
        train_dataset=vl_train,
        eval_dataset=vl_eval,
        max_seq_length=max_seq_length,
        # 注意：此处不传 dataset_text_field
        # 因为数据集已经过 preprocess_vision_language_sample() 预处理，
        # 直接包含了 input_ids、labels 等字段，无需 SFTTrainer 内部再做分词
    )

    # 开始训练（矿物识别阶段必须执行）
    trainer2.train()


def save_model(model, tokenizer, output_dir: str):
    """
    以多种格式保存训练后的模型。

    保存三个版本：
      1. lora_adapter  —— 仅保存 LoRA 适配器权重（文件最小，需要配合原始模型使用）
      2. merged_16bit  —— 将 LoRA 权重合并到原始模型中，保存为 16-bit 精度（可直接使用）
      3. gguf          —— 量化为 GGUF 格式（用于 llama.cpp 等本地推理框架，q4_k_m=4-bit 量化）

    参数：
        model      : 训练后的模型
        tokenizer  : 分词器
        output_dir (str): 模型保存根目录
    """
    print("\n" + "=" * 50)
    print("  保存模型")
    print("=" * 50)

    lora_path   = os.path.join(output_dir, "lora_adapter")
    merged_path = os.path.join(output_dir, "merged_16bit")
    gguf_path   = os.path.join(output_dir, "gguf")

    # 保存 LoRA 适配器（仅包含微调新增的参数，体积小）
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    # 将 LoRA 权重合并回原始模型，保存为完整的 16-bit 模型（可直接加载推理）
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    # 导出为 GGUF 格式，使用 Q4_K_M 量化（适合在 CPU/本地设备上运行）
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")

    print(f"\nDone! Saved to: {output_dir}")


# ==============================================================================
# 主程序入口
# 注：Python 中 "if __name__ == '__main__':" 相当于 Java 的 main() 方法
#     当此脚本被直接运行时执行，被其他脚本 import 时不执行
# ==============================================================================

if __name__ == "__main__":

    # 图片数据目录（独立于 DATA_DIR，单独定义是因为路径不同）
    IMG_DIR = r"D:\OneDrive\Desktop\train\img"
    # 注：r"..." 是 raw string，反斜杠不会被当作转义符（Python 特有语法）

    # ------------------------------------------------------------------
    # Step 1：加载文本问答数据集
    # ------------------------------------------------------------------
    text_ds = load_text_qa_dataset(DATA_DIR)

    # ------------------------------------------------------------------
    # Step 2：加载矿物图片识别数据集
    # ------------------------------------------------------------------
    vl_splits = load_mineral_image_dataset(IMG_DIR)

    # ------------------------------------------------------------------
    # Step 3：加载预训练模型并附加 LoRA 适配器
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_apply_lora(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
    )

    # ------------------------------------------------------------------
    # Step 4：对图片数据集进行预处理（编码为张量）
    # ------------------------------------------------------------------
    vl_train, vl_eval = prepare_vision_language_dataset(
        vl_splits=vl_splits,
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # ------------------------------------------------------------------
    # Step 5：训练阶段一 —— 文本问答
    # （注意：trainer1.train() 在函数内部被注释掉，当前不会实际训练）
    # ------------------------------------------------------------------
    train_text_qa(
        model=model,
        tokenizer=tokenizer,
        text_ds=text_ds,
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # ------------------------------------------------------------------
    # Step 6：训练阶段二 —— 矿物图片识别
    # ------------------------------------------------------------------
    train_mineral_recognition(
        model=model,
        tokenizer=tokenizer,
        vl_train=vl_train,
        vl_eval=vl_eval,
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # ------------------------------------------------------------------
    # Step 7：保存最终模型
    # ------------------------------------------------------------------
    save_model(model, tokenizer, OUTPUT_DIR)
