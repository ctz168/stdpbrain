#!/bin/bash

# 类人脑AI架构 - Conda 环境配置脚本
# 适用于 macOS/Linux

set -e

echo "============================================================"
echo "类人脑双系统全闭环 AI架构 - 环境配置"
echo "============================================================"

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 未检测到 conda"
    echo ""
    echo "请先安装 Miniconda:"
    echo "  macOS: brew install --cask miniconda"
    echo "  或从 https://docs.conda.io/en/latest/miniconda.html 下载"
    exit 1
fi

echo "✓ conda 已安装"

# 设置环境名称
ENV_NAME="${1:-stdpbrain}"
PYTHON_VERSION="${2:-3.11}"

echo ""
echo "创建 conda 环境：$ENV_NAME (Python $PYTHON_VERSION)"
echo ""

# 删除已存在的环境
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  环境已存在，正在删除..."
    conda remove --name $ENV_NAME --all -y
fi

# 创建新环境
echo "📦 创建 Python $PYTHON_VERSION 环境..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 激活环境
echo ""
echo "📦 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 安装 PyTorch
echo ""
echo "📦 安装 PyTorch (CPU 版本)..."
conda install pytorch cpuonly -c pytorch -y

# 验证 PyTorch 安装
python -c "import torch; print(f'✓ PyTorch {torch.__version__} 安装成功')"

# 安装 Transformers 和其他依赖
echo ""
echo "📦 安装 Transformers 和依赖..."
pip install transformers sentencepiece accelerate optimum

# 安装 Telegram Bot 依赖
echo ""
echo "📦 安装 Telegram Bot 依赖..."
pip install python-telegram-bot aiohttp

# 安装其他工具
echo ""
echo "📦 安装其他工具..."
pip install numpy scipy scikit-learn pandas tqdm

# 验证安装
echo ""
echo "============================================================"
echo "验证安装"
echo "============================================================"

python << 'EOF'
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    from transformers import AutoTokenizer
    print("✓ Transformers")
except ImportError as e:
    print(f"✗ Transformers: {e}")

try:
    import telegram
    print(f"✓ python-telegram-bot {telegram.__version__}")
except ImportError as e:
    print(f"✗ python-telegram-bot: {e}")

print("\n✅ 基础依赖安装完成!")
EOF

# 显示使用说明
echo ""
echo "============================================================"
echo "✅ 环境配置完成!"
echo "============================================================"
echo ""
echo "使用方法:"
echo "  1. 激活环境：conda activate $ENV_NAME"
echo "  2. 测试简化版：python test_run.py"
echo "  3. 测试完整版：python main.py --mode chat"
echo "  4. Telegram Bot: python main.py --mode telegram"
echo ""
echo "提示:"
echo "  - 如需 GPU 支持：conda install pytorch torchvision cudatoolkit -c pytorch"
echo "  - 查看文档：cat RUN_INSTRUCTIONS.md"
echo ""
