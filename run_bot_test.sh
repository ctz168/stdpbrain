#!/bin/bash
# 一键启动类人脑 AI Bot 测试环境 (模拟模式)

# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:.

# 获取环境中的 python 路径
PYTHON_BIN="/opt/anaconda3/envs/stdpbrain/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ 找不到 python 环境: $PYTHON_BIN"
    exit 1
fi

echo "============================================================"
echo "🧠 类人脑双系统全闭环 AI 架构 - Bot 测试启动中"
echo "模式：模拟模式 (无需 Telegram Token)"
echo "功能：后台自思考流、SWR 记忆巩固、流式输出、STDP 学习"
echo "============================================================"

# 运行测试脚本
$PYTHON_BIN tests/test_bot_new.py

echo "============================================================"
echo "✨ 测试完成！如果需要运行真实的 Telegram Bot，请使用："
echo "export PYTHONPATH=\$PYTHONPATH:. && $PYTHON_BIN main.py --mode telegram --telegram-token YOUR_TOKEN"
echo "============================================================"
