# 快速使用指南

## 5 分钟快速开始

### 步骤 1: 安装依赖 (2 分钟)

```bash
cd /Users/hilbert/Desktop/stdpbrian
pip install -r requirements.txt
```

### 步骤 2: 下载模型 (取决于网络速度)

```bash
python -m huggingface_hub.cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base

# 使用 ModelScope (国内推荐)
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B-Base', local_dir='./models/Qwen3.5-0.8B-Base')"
```

### 步骤 3: 运行第一个示例

**对话模式:**
```bash
python main.py --mode chat
```

输入任意问题，例如:
```
你：介绍一下类人脑AI架构
AI: [回复]
```

**单次生成:**
```bash
python main.py --mode generate --input "请用 100 字解释什么是 STDP"
```

**查看系统统计:**
```bash
python main.py --mode stats
```

**Telegram Bot 模式 (新增):**
```bash
# 安装额外依赖
pip install python-telegram-bot aiohttp

# 启动 Bot
python main.py --mode telegram
```

然后在 Telegram 中搜索你的 Bot 用户名，发送 `/start` 开始对话！

---

## 常用操作

### 1. 查看配置

```python
from configs.arch_config import default_config

print(f"刷新周期：{default_config.hard_constraints.REFRESH_PERIOD_MS}ms")
print(f"最大显存：{default_config.hard_constraints.MAX_MEMORY_MB}MB")
print(f"STDP 学习率：α={default_config.stdp.alpha_LTP}, β={default_config.stdp.beta_LTD}")
```

### 2. 自定义配置

```python
from configs.arch_config import BrainAIConfig

config = BrainAIConfig()
config.hippocampus.CA3_max_capacity = 5000  # 减少记忆容量以节省内存
config.self_loop.mode3_eval_period = 5     # 更频繁的自评判

# 使用自定义配置创建 AI
from core.interfaces import BrainAIInterface
ai = BrainAIInterface(config)
```

### 3. 批量生成

```python
from core.interfaces import create_brain_ai

ai = create_brain_ai()

prompts = [
    "什么是人工智能？",
    "请写一首关于春天的诗",
    "解释量子力学的基本原理"
]

for prompt in prompts:
    output = ai.generate(prompt, max_tokens=100)
    print(f"\n输入：{prompt}")
    print(f"输出：{output.text}")
    print(f"置信度：{output.confidence:.2%}")
```

### 4. 多轮对话

```python
ai = create_brain_ai()

history = []

while True:
    user_input = input("你：")
    if user_input.lower() == 'quit':
        break
    
    response = ai.chat(user_input, history)
    print(f"AI: {response}")
    
    # 更新历史
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
```

### 5. 性能测试

```python
import time
from core.interfaces import create_brain_ai

ai = create_brain_ai()

# 测试延迟
prompts = ["你好"] * 10
latencies = []

for prompt in prompts:
    start = time.time()
    ai.generate(prompt, max_tokens=50)
    latency = (time.time() - start) * 1000
    latencies.append(latency)

print(f"平均延迟：{sum(latencies)/len(latencies):.2f}ms")
print(f"P95 延迟：{sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
```

### 6. 保存与加载

```python
# 保存检查点
ai.save_checkpoint("./checkpoints/latest.pt")

# 加载检查点
ai.load_checkpoint("./checkpoints/latest.pt")
```

---

## 故障排查

### 问题 1: 找不到模块

错误信息:
```
ModuleNotFoundError: No module named 'xxx'
```

解决:
```bash
pip install -r requirements.txt
```

### 问题 2: 模型未找到

错误信息:
```
FileNotFoundError: [Errno 2] No such file or directory: './models/Qwen3.5-0.8B-Base'
```

解决:
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B-Base', local_dir='./models/Qwen3.5-0.8B-Base')"
```

### 问题 3: CUDA out of memory

错误信息:
```
RuntimeError: CUDA out of memory
```

解决:
```bash
# 使用 CPU 运行
python main.py --mode chat --device cpu

# 或启用 INT4 量化
export QUANTIZATION=INT4
```

### 问题 4: 导入错误

错误信息:
```
ImportError: cannot import name 'xxx' from 'core'
```

解决:
```bash
# 确保在项目根目录运行
cd /Users/hilbert/Desktop/stdpbrian
python main.py
```

### 问题 5: Telegram Bot 无法启动 (新增)

**错误**: `No module named 'telegram'`

解决:
```bash
pip install python-telegram-bot aiohttp
```

**错误**: `Unauthorized`

解决：检查 Bot Token 是否正确
```bash
python telegram_bot/run.py --token YOUR_BOT_TOKEN
```

---

## 进阶使用

### 1. 调整 STDP 参数

```python
from configs.arch_config import BrainAIConfig

config = BrainAIConfig()
config.stdp.alpha_LTP = 0.015  # 增强学习率
config.stdp.beta_LTD = 0.010   # 减弱学习率
config.stdp.time_window_ms = 25  # 延长时问窗口

ai = BrainAIInterface(config)
```

### 2. 调整海马体容量

```python
config.hippocampus.CA3_max_capacity = 20000  # 增加记忆容量
config.hippocampus.EC_feature_dim = 128      # 增加编码维度
```

### 3. 调整自闭环策略

```python
config.self_loop.mode1_temperature_range = (0.8, 1.0)  # 提高温度
config.self_loop.mode2_max_iterations = 10             # 更多迭代
config.self_loop.mode3_eval_period = 5                 # 更频繁评判
```

---

## 获取帮助

### 查看完整文档

- [README.md](README.md) - 项目说明
- [ARCHITECTURE.md](ARCHITECTURE.md) - 架构设计
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 交付总结
- [deployment/README.md](deployment/README.md) - 部署指南
- **[TELEGRAM_BOT_SUMMARY.md](TELEGRAM_BOT_SUMMARY.md)** - Telegram Bot 功能总结 ⭐ **新增**

### Telegram Bot 帮助 (新增)

- **[INSTALL_TELEGRAM.md](INSTALL_TELEGRAM.md)** - Bot 安装说明
- **[telegram_bot/README.md](telegram_bot/README.md)** - Bot 详细使用指南
- **[telegram_bot/test_bot.py](telegram_bot/test_bot.py)** - 测试 Bot 功能

### 运行测试

```bash
pytest tests/ -v

# Telegram Bot 测试
python telegram_bot/test_bot.py
```

### 查看日志

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
python main.py --mode chat
```

---

*最后更新：2026-03-09*
