# 快速使用指南

## ⚠️ 重要提示

**Python 版本**: 必须使用 **Python3.11** (PyTorch 兼容性)

---

## 5 分钟快速开始

### 步骤 1: 准备环境（2 分钟）

**方法 A: 使用 Conda（推荐）**

```bash
cd /Users/hilbert/Desktop/stdpbrian

# 创建并激活环境
conda create -n stdpbrain python=3.11 -y
conda activate stdpbrain

# 一键安装所有依赖
./setup_conda_env.sh
```

**方法 B: 手动安装**

```bash
# 创建 Python3.11 环境
conda create -n stdpbrain python=3.11
conda activate stdpbrain

# 安装 PyTorch (CPU 版本)
conda install pytorch cpuonly -c pytorch -y

# 安装其他依赖
pip install transformers sentencepiece accelerate optimum
pip install python-telegram-bot aiohttp
pip install numpy scipy scikit-learn pandas tqdm pyyaml
```

### 步骤 2: 验证环境（30 秒）

```bash
python check_env.py
```

应看到：
```
============================================================
类人脑 AI 架构 - 环境验证
============================================================

Python: 3.11
✓ Python 版本正确
PyTorch: 2.5.1 ✓
Transformers: 5.3.0 ✓
模型目录：✓
Qwen 接口：✓

============================================================
✅ 环境验证完成!
============================================================
```

### 步骤 3: 下载模型（取决于网络）

**从 HuggingFace 下载:**
```bash
huggingface-cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base
```

**从 ModelScope 下载（国内推荐）:**
```bash
pip install modelscope
modelscope download Qwen/Qwen3.5-0.8B-Base --local_dir ./models/Qwen3.5-0.8B-Base
```

**检查模型文件:**
```bash
ls models/Qwen3.5-0.8B-Base/
# 应包含：config.json, tokenizer.json, model.safetensors 等
```

### 步骤 4: 运行第一个示例（1 分钟）

**简单测试（推荐首次运行）:**
```bash
python simple_test.py
```

**完整功能测试:**
```bash
python final_test.py
```

**对话模式:**
```bash
python main.py --mode chat
```

输入问题，例如：
```
你：你好，请介绍一下自己
AI: [回复]
```

**单次生成:**
```bash
python main.py --mode generate --input "请用 50 字解释人工智能"
```

**查看统计:**
```bash
python main.py --mode stats
```

---

## Telegram Bot 快速开始（新增）

### 1. 启动 Bot

```bash
python main.py --mode telegram --telegram-token YOUR_BOT_TOKEN
```

或使用默认 Token（已配置）:
```bash
python main.py --mode telegram
```

### 2. 在 Telegram 中使用

1. 打开 Telegram
2. 搜索你的 Bot 用户名
3. 发送 `/start` 开始对话
4. 发送任意消息进行交互

### 3. 可用命令

- `/start` - 开始对话
- `/help` - 显示帮助信息
- `/clear` - 清除对话历史
- `/stats` - 查看系统统计

---

## 常用操作示例

### 1. 基础对话

```python
from core.qwen_interface import create_qwen_ai

ai = create_qwen_ai(model_path="./models/Qwen3.5-0.8B-Base")
response = ai.chat("你好")
print(response)
```

### 2. 文本生成

```python
# 续写
text = ai.generate("人工智能的核心是", max_new_tokens=100)
print(text)

# 创作
poem = ai.generate("写一首关于春天的诗", max_new_tokens=200)
print(poem)
```

### 3. 多轮对话

```python
history = []

while True:
   user_input = input("你：")
   if user_input.lower() == 'quit':
        break
    
   response = ai.chat(user_input, history)
   print(f"AI: {response}")
    
    # 更新历史（保留最近 5 轮）
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
```

### 4. 查看统计

```python
stats = ai.get_stats()
print(f"生成次数：{stats['generation_count']}")
print(f"总 token 数：{stats['total_tokens']}")
print(f"设备：{stats['device']}")

if 'hippocampus' in stats:
    hp = stats['hippocampus']
  print(f"海马体记忆数：{hp.get('num_memories', 0)}")
  print(f"内存使用：{hp.get('memory_usage_mb', 0):.2f} MB")
```

---

## 故障排查

### 问题 1: Python 版本错误

**错误信息:**
```
ModuleNotFoundError: No module named 'torch'
```

**解决:**
```bash
# 检查 Python 版本
python --version  # 应该是 3.11.x

# 如果不对，创建正确版本的环境
conda create-n stdpbrain python=3.11
conda activate stdpbrain
```

### 问题 2: 模型未找到

**错误信息:**
```
FileNotFoundError: Model not found at ./models/Qwen3.5-0.8B-Base
```

**解决:**
```bash
# 检查模型目录
ls models/Qwen3.5-0.8B-Base/

# 重新下载
huggingface-cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base
```

### 问题 3: NumPy 版本冲突

**错误信息:**
```
ValueError: numpy.dtype size changed
```

**解决:**
```bash
pip install "numpy<2.0"
```

### 问题 4: Transformers 版本太低

**错误信息:**
```
Qwen3.5 not supported
```

**解决:**
```bash
pip install --upgrade transformers
```

### 问题 5: Telegram Bot 无法启动

**错误:** `No module named 'telegram'`

**解决:**
```bash
pip install python-telegram-bot aiohttp
```

**错误:** `Unauthorized`

**解决:** 检查 Bot Token 是否正确
```bash
python main.py --mode telegram --telegram-token YOUR_VALID_TOKEN
```

---

## 性能优化建议

### CPU 模式（当前默认）
- 生成速度：~3-4 tokens/s
- 内存占用：~3GB
- 适合：测试、开发

### GPU 模式（推荐）
```bash
# 安装 CUDA 版 PyTorch
conda install pytorch torchvision cudatoolkit -c pytorch

# 使用 GPU 运行
python main.py --mode chat --device cuda
```
预期性能：
- 生成速度：30-50 tokens/s
- 内存占用：~1.5GB

### INT4 量化（高级）
```python
ai = create_qwen_ai(
   model_path="./models/Qwen3.5-0.8B-Base",
   use_int4=True  # 启用量化
)
```
- 内存减少：~4x
- 速度提升：~2x

---

## 获取帮助

### 文档
- **[RUN_GUIDE.md](RUN_GUIDE.md)** - 详细运行指南 ⭐
- **[TEST_REPORT.md](TEST_REPORT.md)** - 测试报告 ⭐
- **[PROJECT_COMPLETION.md](PROJECT_COMPLETION.md)** - 完成总结 ⭐
- **[README.md](README.md)** - 项目说明
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 架构详解

### 快速参考
```bash
# 环境验证
python check_env.py

# 简单测试
python simple_test.py

# 查看文档
cat RUN_GUIDE.md
cat QUICK_REFERENCE.md
```

### 测试脚本
```bash
# 完整测试
python final_test.py

# Telegram Bot 测试
python telegram_bot/test_bot.py
```

---

## 下一步

1. ✅ 完成快速开始
2. 📖 阅读 [RUN_GUIDE.md](RUN_GUIDE.md) 了解更多用法
3. 🧪 运行 [final_test.py](file:///Users/hilbert/Desktop/stdpbrian/final_test.py) 测试完整功能
4. 💬 尝试 Telegram Bot 交互
5. 🔧 根据需求调整配置

---

*最后更新：2026-03-09*  
*适用版本：v1.0*  
*Python: 3.11+ | PyTorch: 2.5+*
