# stdpbrain

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ctz168/stdpbrain/blob/main/colab_stdpbrain_bot.ipynb)

</div>

> 说明：本 README 只保留可从当前代码直接对齐的信息；不再包含未经可重复实验验证的"指标/提升百分比/实测性能表"。

一个基于 **Qwen3.5-2B** 的实验性双系统架构项目：
- 新皮层侧：`core/qwen_interface.py` + `core/qwen_narrow_band_patch.py`
- 海马体侧：`hippocampus/`（EC / DG / CA3 / CA1 / SWR）
- 在线可塑性：`core/stdp_engine.py`
- 统一接口：`core/interfaces.py`

## 项目结构
```text
stdpbrain/
├── main.py                         # CLI 入口(chat/generate/eval/stats/telegram/...)
├── config.py                       # 运行时配置(默认本地模型路径)
├── configs/arch_config.py          # 架构参数(HardConstraints/STDP/Hippocampus/...)
├── core/
│   ├── interfaces.py               # BrainAIInterface
│   ├── qwen_interface.py           # Qwen 模型封装
│   ├── qwen_narrow_band_patch.py   # 注意力补丁(窄带宽 / KV 压缩相关)
│   └── stdp_engine.py              # STDP 引擎
├── hippocampus/
│   ├── hippocampus_system.py       # 海马体系统协调
│   ├── ca3_memory.py               # 情景记忆存取与召回
│   ├── dg_separator.py             # 模式分离
│   ├── ca1_gate.py                 # 注意力门控
│   └── swr_consolidation.py        # 回放巩固
└── tests/
    └── test_p0_recall_and_rope.py  # RoPE/召回回归测试
```

## 环境要求

- Python 3.11+(仓库中的环境脚本与当前代码实践基于 3.11)
- PyTorch 2.x
- 依赖见 `requirements.txt`

安装示例:

```bash
pip install -r requirements.txt
```

## 模型准备(本地)

当前默认模型路径:

```python
MODEL_PATH = "./models/Qwen3.5-2B"
```

请先准备本地模型目录(含 tokenizer/config/weights),再运行 `main.py`。

## 运行方式

```bash
# 对话模式
python main.py --mode chat

# 持续模式
python main.py --mode continuous

# 生成模式
python main.py --mode generate --input "请解释量子力学"

# 评测模式
python main.py --mode eval

# 统计模式
python main.py --mode stats

# Telegram Bot
python main.py --mode telegram --telegram-token <YOUR_TOKEN>
```

## 已有测试

```bash
python -m unittest tests/test_p0_recall_and_rope.py
```

该测试覆盖:
- KV 压缩后 window 区域不被二次 RoPE 处理(回归保护)
- CA3 召回 trace 的索引/语义分数对齐

## 配置入口

- 基础运行配置:`config.py`
- 架构参数配置:`configs/arch_config.py`

可优先调整项:
- `BrainAIConfig.model_path`
- `HardConstraints`(窗口、刷新周期、KV 相关开关)
- `HippocampusConfig`(召回阈值、topk、容量)
- `STDPConfig`(学习率、阈值、衰减)

## 声明

本项目处于持续迭代阶段。README 不再承诺固定性能指标;建议以你当前环境下可复现的测试日志与评估结果为准。
