#!/usr/bin/env python
"""环境验证脚本"""
import sys, os

print("=" * 60)
print("类人脑 AI 架构 - 环境验证")
print("=" * 60)

# Python 版本
print(f"\nPython: {sys.version_info.major}.{sys.version_info.minor}")
assert sys.version_info >= (3, 11), "需要 Python>=3.11"
print("✓ Python 版本正确")

# PyTorch
try:
   import torch
   print(f"PyTorch: {torch.__version__} ✓")
except:
   print("PyTorch: ✗")

# Transformers
try:
   import transformers
   print(f"Transformers: {transformers.__version__} ✓")
except:
   print("Transformers: ✗")

# 模型
model_path = "./models/Qwen3.5-0.8B-Base"
if os.path.exists(model_path):
   print(f"模型目录：✓")
else:
   print(f"模型目录：✗")

# Qwen 接口
try:
   from core.qwen_interface import create_qwen_ai
   print("Qwen 接口：✓")
except Exception as e:
   print(f"Qwen 接口：✗ ({e})")

print("\n" + "=" * 60)
print("✅ 环境验证完成!")
print("=" * 60)
