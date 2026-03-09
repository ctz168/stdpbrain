# 类人脑 AI 架构 - 运行指南

## 快速开始

### 1. 激活 Conda 环境
```bash
conda activate stdpbrain
```

### 2. 测试系统
```bash
# 简单测试（推荐首次运行）
python simple_test.py

# 完整功能测试
python final_test.py
```

### 3. 对话模式
```bash
python main.py --mode chat
```

### 4. Telegram Bot
```bash
python main.py --mode telegram --telegram-token YOUR_BOT_TOKEN
```

## 项目结构

```
stdpbrian/
├── core/                      # 核心模块
│   ├── qwen_interface.py    # Qwen 模型接口（真实模型）
│   ├── dual_weight_layers.py # 双权重层
│   ├── stdp_engine.py       # STDP 学习引擎
│   ├── refresh_engine.py    # 100Hz刷新引擎
│   └── interfaces.py        # AI 接口
├── hippocampus/              # 海马体系统
│   ├── ec_cortex.py        # 内嗅皮层编码器
│   ├── dg_separator.py     # 齿状回分离器
│   ├── ca3_memory.py       # CA3 记忆存储
│   ├── ca1_gate.py         # CA1 门控
│   └── swr_consolidation.py # SWR 记忆巩固
├── self_loop/                # 自闭环优化
├── telegram_bot/             # Telegram 机器人
├── configs/                  # 配置文件
│   └── arch_config.py      # 架构配置
├── models/                   # 模型文件
│   └── Qwen3.5-0.8B-Base/   # Qwen 模型
├── main.py                  # 主入口
├── test_run.py             # 简化测试（无依赖）
├── simple_test.py          # 简单测试（有依赖）
└── final_test.py           # 完整测试
```

## 核心组件说明

### 1. Qwen 真实模型接口 (`core/qwen_interface.py`)

**功能**:
- 加载真实的 Qwen3.5-0.8B 模型
- 使用官方 tokenizer
- 支持文本生成和对话
- 集成海马体系统

**使用示例**:
```python
from core.qwen_interface import create_qwen_ai

ai = create_qwen_ai(
  model_path="./models/Qwen3.5-0.8B-Base",
  device="cpu"
)

# 对话
response = ai.chat("你好")
print(response)

# 生成
text = ai.generate("人工智能", max_new_tokens=100)
```

### 2. 海马体系统 (`hippocampus/`)

**5 个子模块**:
1. **EC (Entorhinal Cortex)**: 感觉信息编码器
2. **DG (Dentate Gyrus)**: 模式分离器
3. **CA3**: 快速记忆存储（索引式）
4. **CA1**: 记忆检索门控
5. **SWR (Sharp Wave Ripple)**: 记忆巩固

**特点**:
- 记忆容量限制：≤2MB
- 支持情景记忆编码
- 自动记忆巩固（后台线程）

### 3. STDP 学习引擎 (`core/stdp_engine.py`)

**学习规则**:
```
Δw = α * exp(-Δt/τ)  if Δt > 0  (LTP 增强)
Δw = -β * exp(Δt/τ) if Δt < 0  (LTD 抑制)
```

**全链路更新**:
- Attention 权重
- FFN 权重
- 自评估权重
- 海马体门控权重

### 4. 双权重层 (`core/dual_weight_layers.py`)

**架构**:
- **90% 静态权重**: 冻结，来自预训练 Qwen 模型
- **10% 动态权重**: 可学习，通过 STDP 在线更新

**优势**:
- 保持预训练知识
- 支持在线学习
- 防止灾难性遗忘

## 性能优化建议

### CPU 模式（当前）
- 生成速度：~3-4 tokens/s
- 内存占用：~3GB
- 适合测试和开发

### GPU 模式（推荐）
```bash
# 安装 CUDA 版 PyTorch
conda install pytorch torchvision cudatoolkit-c pytorch

# 运行
python main.py --mode chat --device cuda
```
预期性能提升：10-20x

### INT4 量化
```python
ai = create_qwen_ai(
  model_path="./models/Qwen3.5-0.8B-Base",
  use_int4=True  # 启用 INT4 量化
)
```
- 内存减少：~4x
- 速度提升：~2x
- 需要 optimum.quanto 库

## 常见问题

### Q1: Python 版本不兼容
**错误**: `ModuleNotFoundError: No module named 'torch'`

**解决**:
```bash
# 必须使用 Python3.11
conda activate stdpbrain
# 或重新创建环境
./setup_conda_env.sh
```

### Q2: 模型加载失败
**错误**: `Model not found at ./models/Qwen3.5-0.8B-Base`

**解决**:
```bash
# 检查模型目录
ls models/Qwen3.5-0.8B-Base/

# 应包含以下文件:
# - config.json
# - tokenizer.json
# - model.safetensors
```

### Q3: 生成速度太慢
**原因**: CPU 模式运行

**解决**:
1. 使用 GPU（见上方 GPU 模式）
2. 使用 Apple Silicon MPS 加速
3. 减少 max_new_tokens 参数

### Q4: Telegram Bot 无法启动
**错误**: `Unauthorized` 或 `Token invalid`

**解决**:
```bash
# 检查 token 是否正确
python main.py --mode telegram --telegram-token YOUR_VALID_TOKEN

# 确保已安装依赖
pip install python-telegram-bot aiohttp
```

## 开发指南

### 添加新功能
1. 在对应模块目录创建文件
2. 在 `__init__.py` 中导出
3. 在 `main.py` 中添加调用

### 修改配置
编辑 `configs/arch_config.py`:
```python
default_config = ArchConfig(
   model_name="Qwen3.5-0.8B-Base",
    # ... 修改其他参数
)
```

### 调试技巧
```python
# 查看详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看统计
stats = ai.get_stats()
print(stats)

# 查看海马体状态
if ai.hippocampus:
  print(ai.hippocampus.get_stats())
```

## 参考资料

- [ARCHITECTURE.md](ARCHITECTURE.md) - 架构详解
- [TEST_REPORT.md](TEST_REPORT.md) - 测试报告
- [README.md](README.md) - 项目说明
- [Qwen3.5 文档](https://qwen.readthedocs.io/)

---

**最后更新**: 2026-03-09  
**维护者**: 类人脑 AI 项目组
