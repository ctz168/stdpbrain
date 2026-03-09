# Python 环境配置指南

## ⚠️ 重要提示

当前系统使用 **Python 3.13**，但 PyTorch 尚未发布对应版本。需要创建 Python 3.11 或 3.10 的虚拟环境。

---

## 🔧 方案 1: 使用 conda (推荐)

### 步骤 1: 安装 Miniconda (如果未安装)

```bash
# macOS
brew install --cask miniconda

# 或从官网下载
# https://docs.conda.io/en/latest/miniconda.html
```

### 步骤 2: 创建 Python 3.11 环境

```bash
cd /Users/hilbert/Desktop/stdpbrian

# 创建环境
conda create -n stdpbrain python=3.11 -y

# 激活环境
conda activate stdpbrain
```

### 步骤 3: 安装依赖

```bash
# 安装 PyTorch (CPU 版本)
conda install pytorch cpuonly -c pytorch -y

# 或使用 pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install transformers sentencepiece accelerate optimum python-telegram-bot aiohttp
```

### 步骤 4: 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully!')"
```

---

## 🔧 方案 2: 使用 pyenv

### 步骤 1: 安装 pyenv

```bash
brew install pyenv
```

### 步骤 2: 安装 Python 3.11

```bash
pyenv install 3.11.9
```

### 步骤 3: 设置项目 Python 版本

```bash
cd /Users/hilbert/Desktop/stdpbrian
pyenv local 3.11.9
```

### 步骤 4: 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate
```

### 步骤 5: 安装依赖

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece accelerate optimum python-telegram-bot aiohttp
```

---

## 🚀 快速脚本 (如果使用 conda)

创建 `setup_env.sh`:

```bash
#!/bin/bash

echo "Creating conda environment..."
conda create -n stdpbrain python=3.11 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stdpbrain

echo "Installing PyTorch (CPU)..."
conda install pytorch cpuonly -c pytorch -y

echo "Installing other dependencies..."
pip install transformers sentencepiece accelerate optimum python-telegram-bot aiohttp

echo "Environment setup complete!"
echo "Activate with: conda activate stdpbrain"
```

运行:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

## ✅ 验证安装

创建 `test_install.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

# 测试 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print("✓ Transformer tokenizer loaded")

# 简单测试
text = "你好"
tokens = tokenizer.encode(text)
print(f"✓ Tokenization test: '{text}' -> {len(tokens)} tokens")

print("\n✅ All dependencies installed successfully!")
```

运行:
```bash
conda activate stdpbrain
python test_install.py
```

---

## 📝 当前模型说明

您提到已经下载了模型，请确认模型路径：

```bash
# 检查模型目录
ls -la ./models/
```

应该看到类似:
```
models/
└── Qwen3.5-0.8B-Base/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── pytorch_model.bin (或 .safetensors)
```

---

## 🎯 下一步

环境配置完成后:

1. **测试简化版** (当前可用):
   ```bash
   python test_run.py
   ```

2. **测试完整版** (安装依赖后):
   ```bash
   # 激活环境
   conda activate stdpbrain
   
   # 运行对话
   python main.py --mode chat
   
   # 或启动 Telegram Bot
   python main.py --mode telegram
   ```

---

## 🔗 相关资源

- [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [Qwen 模型下载](https://huggingface.co/Qwen)

---

*最后更新：2026-03-09*
