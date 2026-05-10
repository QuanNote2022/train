# 矿物识别助手 — Qwen3.5-2B 微调

基于 Qwen3.5-2B 多模态模型，使用 Unsloth QLoRA 微调的矿物识别助手，支持文本问答和图片检测。

## 功能

- **文本问答**：矿物知识科普、鉴定方法、产地、收藏养护等
- **图片检测**：矿物图片识别，输出 JSON 格式结果（矿物名、化学式、硬度、光泽等）
- **mineralContext 问答**：基于已识别矿物信息的上下文追问

## 环境

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 (6GB VRAM) |
| Python | 3.11 |
| 微调框架 | Unsloth + QLoRA (4-bit) |
| 基座模型 | Qwen3.5-2B (多模态) |
| 训练显存 | ~5-6 GB |

## 快速开始

### 安装依赖

```bash
conda create -n mineral_lora python=3.11
conda activate mineral_lora
pip install torch unsloth transformers datasets peft trl accelerate bitsandbytes
```

### 训练

```bash
python train_qwen35_unified.py
```

### 推理测试

```bash
python test_mineral_model.py                  # 预设问题测试
python test_mineral_model.py --interactive    # 交互模式
```

交互模式下 `@图片路径` 加载图片检测，直接输入文字进行问答。

## 项目结构

```
train/
├── train_qwen35_unified.py   # 统一训练脚本（文本 + 图片两阶段）
├── test_mineral_model.py     # 推理测试脚本
├── data/
│   ├── train.json            # 文本问答训练集
│   └── eval.json             # 文本问答验证集
├── doc/
│   └── TRAINING_GUIDE.md     # 详细训练指南
└── mineral_unified_model/    # 训练产物（LoRA / 合并模型 / GGUF）
```

## 训练配置

| 参数 | 值 |
|------|-----|
| LoRA rank / alpha | 16 / 16 |
| 量化 | 4-bit (bitsandbytes) |
| 学习率 | 2e-4 |
| 调度器 | cosine |
| 优化器 | adamw_8bit |
| 最大步数 | 180 (~3.9 epochs) |
| 精度 | fp16 |

## 部署

合并模型可导出为 GGUF 格式，通过 Ollama 部署为 OpenAI-compatible API：

```bash
ollama create mineral-assistant -f Modelfile
ollama serve
```

## 数据集

- 文本问答：高质量中文矿物知识问答（概念、鉴定、产地、收藏等）
- 图片检测：[mineralimage5K-98](https://huggingface.co/datasets/Nech-C/mineralimage5K-98)，筛选 20 种常见矿物

## 20 种训练矿物

石英、紫水晶、黄铁矿、方解石、萤石、赤铁矿、磁铁矿、孔雀石、蓝铜矿、石膏、方铅矿、辰砂、自然金、自然铜、硫磺、绿柱石、刚玉、黄玉、电气石、闪锌矿
