# Qwen3.5-2B 矿物识别助手 微调方案

## 概述

基于 Qwen3.5-2B 多模态模型（`image-text-to-text`），使用 Unsloth QLoRA 进行微调，同时支持：

- **文本问答**：矿物知识科普、mineralContext 上下文问答、外观描述鉴定
- **图片检测**：矿物图片识别，输出 JSON 格式的检测结果（含 bbox 边界框）

训练后通过 OpenAI-compatible API 部署（Ollama / llama.cpp），与 `mineral-system/backend` 无缝对接。

---

## 环境

| 项目 | 详情 |
|------|------|
| 操作系统 | Windows 11 Pro |
| GPU | NVIDIA GeForce RTX 3060 (6GB VRAM) |
| CUDA | 13.2 (驱动), 12.4 (PyTorch) |
| Python | 3.11 (conda: `mineral_lora`) |
| 微调框架 | Unsloth 2026.4.8 |
| PyTorch | 2.6.0+cu124 |

### 依赖

```
torch, unsloth, transformers, datasets, peft, trl, 
accelerate, bitsandbytes, xformers, pyarrow
```

---

## 模型

| 项目 | 详情 |
|------|------|
| 基座模型 | `Qwen/Qwen3.5-2B` |
| 模型类型 | 多模态 (`image-text-to-text`, `Qwen3_5ForConditionalGeneration`) |
| 处理器 | `Qwen3VLProcessor` |
| 微调方式 | 4-bit QLoRA (r=16, alpha=16) |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 显存占用 | ~5-6 GB |

---

## 数据集

### 文本问答 (train.json / eval.json)

全部为高质量中文数据，已清除英文混杂样本。

| 类型 | 训练 | 验证 | 示例 |
|------|------|------|------|
| 通用矿物 QA | 86 | ~10 | "什么是莫氏硬度？"、"石英和方解石怎么区分？"、"怎样用简单工具鉴别矿物？" |
| mineralContext | 10 | ~1 | "这个矿物有什么用途？"（带矿物属性上下文，模拟识别后咨询） |
| **文本合计** | **96** | **11** | |

覆盖主题：矿物基础概念、鉴定方法、常见矿物详解、宝石知识、矿物成因分类、中国矿产产地、矿物收藏养护、硬度/光泽/条痕/解理等鉴定特征。

数据格式（Alpaca → ChatML）：

```json
{
  "system": "你是一个专业的矿物识别助手...",
  "instruction": "什么是莫氏硬度？",
  "input": "",
  "output": "莫氏硬度是衡量矿物抵抗刮擦能力的一种标准..."
}
```

### 图片检测 (运行时从 HuggingFace 加载)

| 项目 | 详情 |
|------|------|
| 数据源 | `Nech-C/mineralimage5K-98` |
| 筛选 | 20 种常见矿物 × 15 张/种 = 300 条 |
| 训练/验证 | 270 / 30 (9:1) |
| 图片尺寸 | 1024×700 左右 |
| bbox 标注 | 归一化 0-1 坐标，label 为通用标签（已替换为真实矿物名） |
| description | 71% 为空，使用预设中文描述填充 |

> **数据质量说明**: 原始数据集的 `description` 和 `mineral_boxes[*].label` 字段为英文通用标签（"a stone"等），训练时已将 label 替换为真实矿物中文名，空描述使用预设中文描述补充。

输出格式（与后端 `MineralService.parseDetectionResponse()` 对齐）：

```json
[{
  "label": "石英",
  "confidence": 0.90,
  "bbox": [184, 177, 673, 710],
  "mineralInfo": {
    "name": "石英",
    "formula": "",
    "hardness": "",
    "luster": "",
    "color": "",
    "description": "石英（quartz）..."
  }
}]
```

### 合并后

| 类型 | 训练 | 验证 |
|------|------|------|
| 文本 + 图片 | 366 | 41 |

---

## 20 种训练矿物

| # | 中文 | 英文 | # | 中文 | 英文 |
|---|------|------|---|------|------|
| 1 | 石英 | quartz | 11 | 方铅矿 | galena |
| 2 | 紫水晶 | amethyst | 12 | 辰砂 | cinnabar |
| 3 | 黄铁矿 | pyrite | 13 | 自然金 | gold |
| 4 | 方解石 | calcite | 14 | 自然铜 | copper |
| 5 | 萤石 | fluorite | 15 | 硫磺 | sulfur |
| 6 | 赤铁矿 | hematite | 16 | 绿柱石 | beryl |
| 7 | 磁铁矿 | magnetite | 17 | 刚玉 | corundum |
| 8 | 孔雀石 | malachite | 18 | 黄玉 | topaz |
| 9 | 蓝铜矿 | azurite | 19 | 电气石 | tourmaline |
| 10 | 石膏 | gypsum | 20 | 闪锌矿 | sphalerite |

---

## 训练配置

| 参数 | 值 |
|------|-----|
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| 量化 | 4-bit (bitsandbytes) |
| Batch size | 1 × 8 gradient accumulation = 8 effective |
| Max sequence length | 1024 |
| Max steps | 180 (~3.9 epochs) |
| Learning rate | 2e-4 |
| Scheduler | cosine |
| Optimizer | adamw_8bit |
| Precision | fp16 |
| Warmup steps | 10 |

---

## 消息格式 (ChatML)

与后端 `ChatService.chatWithOllama()` 一致：

```
<|im_start|>system
{system_prompt + mineralContext/ragContext}
<|im_end|>
<|im_start|>user
{user_question}
<|im_end|>
<|im_start|>assistant
{response}
<|im_end|>
```

### 后端消息流对应

```java
// ChatService.chatWithOllama()
messages.add(SystemMessage.from(buildSystemPrompt(ragContext)));  // <|im_start|>system ...
messages.add(UserMessage.from(content));                          // <|im_start|>user ...
// AiMessage.from(response)                                       // <|im_start|>assistant ...
```

---

## 训练

### 命令

```bash
conda activate mineral_lora
python train_qwen35_unified.py
```

首次运行会下载 Qwen3.5-2B (~4GB) 和 mineralimage5K-98 数据集 (~1GB)，之后开始训练。

### 输出

```
mineral_unified_model/
├── lora_adapter/      ← LoRA 权重（可继续增量训练）
├── merged_16bit/      ← 合并完整模型（16-bit，用于部署）
└── gguf/              ← Q4_K_M 量化（用于 Ollama 部署）
```

---

## 推理测试

### 命令

```bash
conda activate mineral_lora
python test_mineral_model.py                  # 预设问题测试
python test_mineral_model.py --interactive    # 交互模式
```

### 交互模式快捷键

| 输入 | 功能 |
|------|------|
| 直接输入文字 | 文本问答 |
| `@图片路径` | 加载图片进行检测 |
| `quit` / `exit` | 退出 |

---

## 后端部署对接

### Ollama 部署

```bash
# 创建 Modelfile
echo "FROM ./mineral_unified_model/gguf/model.gguf" > Modelfile
ollama create mineral-assistant -f Modelfile
ollama serve
```

### 修改后端配置

```yaml
# mineral-system/backend/src/main/resources/application.yml
dashscope:
  base-url: http://127.0.0.1:11434/v1
  api-key: ollama
  model-name: mineral-assistant
  vision-model-name: mineral-assistant
```

### 无需修改后端代码

LangChain4j 的 `OpenAiChatModel` / `OpenAiStreamingChatModel` 原生兼容 OpenAI-compatible API，无需任何代码改动。

| 后端调用 | 对应模型能力 |
|----------|------------|
| `ChatService.chatWithOllama()` | 文本问答（SSE 流式返回） |
| `MineralService.detectMineral()` | 图片检测（JSON + bbox） |

---

## 项目文件结构

```
D:\OneDrive\Desktop\train\
├── doc/
│   └── TRAINING_GUIDE.md     ← 本文档
├── data/
│   ├── train.json            ← 文本问答训练集 (162条)
│   └── eval.json             ← 文本问答验证集 (18条)
├── train_qwen35_unified.py   ← 训练脚本
├── test_mineral_model.py     ← 推理测试脚本
└── show_plan.py              ← 方案展示脚本
```
