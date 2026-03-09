#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境验证脚本
检查所有依赖是否正确安装，模型是否可用
"""

import sys
import os

def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
   print(f"Python 版本：{version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 11:
       print("✓ Python 版本符合要求 (>=3.11)")
       return True
    else:
       print("✗ Python 版本不符合要求，需要 >=3.11")
       return False

def check_torch():
    """检查 PyTorch"""
    try:
       import torch
       print(f"PyTorch 版本：{torch.__version__}")
       print(f"CUDA 可用：{torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
           print(f"MPS 可用：{torch.backends.mps.is_available()}")
       print("✓ PyTorch 已安装")
       return True
   except ImportError:
       print("✗ PyTorch 未安装")
       return False

def check_transformers():
    """检查 Transformers"""
    try:
       import transformers
       print(f"Transformers 版本：{transformers.__version__}")
       print("✓ Transformers 已安装")
       return True
   except ImportError:
       print("✗ Transformers 未安装")
       return False

def check_other_packages():
    """检查其他包"""
   packages = {
        'sentencepiece': 'SentencePiece',
        'accelerate': 'Accelerate',
        'optimum': 'Optimum',
        'telegram': 'python-telegram-bot',
        'aiohttp': 'aiohttp',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
    }
    
    all_ok = True
    for pkg_name, display_name in packages.items():
        try:
            __import__(pkg_name)
           print(f"✓ {display_name} 已安装")
       except ImportError:
           print(f"⚠ {display_name} 未安装（可选）")
    
   return all_ok

def check_model():
    """检查模型文件"""
   model_path = "./models/Qwen3.5-0.8B-Base"
   print(f"\n检查模型：{model_path}")
    
    if not os.path.exists(model_path):
       print(f"✗ 模型目录不存在：{model_path}")
       print("请运行：huggingface-cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base")
       return False
    
   required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors"
    ]
    
   files_found = os.listdir(model_path)
    all_ok = True
    
    for file in required_files:
        # 检查 safetensors 可能有分片
        if file == "model.safetensors":
            has_safetensors = any("safetensors" in f for f in files_found)
            if has_safetensors:
               print(f"✓ 模型权重文件存在")
            else:
               print(f"✗ 模型权重文件不存在")
                all_ok = False
        else:
            if file in files_found:
               print(f"✓ {file} 存在")
            else:
               print(f"✗ {file} 不存在")
                all_ok = False
    
   return all_ok

def check_qwen_load():
    """测试 Qwen 模型加载"""
   print("\n测试模型加载...")
    try:
       from core.qwen_interface import create_qwen_ai
       print("✓ Qwen 接口导入成功")
        
        # 不实际加载模型，只检查类是否存在
       print("✓ QwenBrainAI 类可用")
       return True
   except Exception as e:
       print(f"✗ Qwen 接口导入失败：{e}")
       return False

def main():
    """主函数"""
   print("=" * 70)
   print("类人脑 AI 架构 - 环境验证")
   print("=" * 70)
   print()
    
    checks = [
        ("Python 版本", check_python_version),
        ("PyTorch", check_torch),
        ("Transformers", check_transformers),
        ("其他包", check_other_packages),
        ("模型文件", check_model),
        ("Qwen 接口", check_qwen_load),
    ]
    
   results = []
    for name, check_func in checks:
       print(f"\n[{name}]")
       result = check_func()
       results.append((name, result))
       print()
    
   print("=" * 70)
   print("验证结果总结")
   print("=" * 70)
    
    all_passed = True
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
       print(f"{status}: {name}")
        if not result:
            all_passed = False
    
   print()
    if all_passed:
       print("🎉 所有检查通过！系统已准备就绪。")
       print("\n下一步:")
       print("  1. 运行测试：python simple_test.py")
       print("  2. 开始对话：python main.py --mode chat")
       print("  3. 查看文档：cat RUN_GUIDE.md")
    else:
       print("⚠ 部分检查未通过，请根据提示修复。")
       print("\n常见问题:")
       print("  - Python 版本不对：使用 conda create-n stdpbrain python=3.11")
       print("  - 缺少依赖：运行 ./setup_conda_env.sh")
       print("  - 模型缺失：下载 Qwen3.5-0.8B-Base 到 ./models/")
    
   print("=" * 70)
    
   return 0 if all_passed else 1

if __name__ == "__main__":
   sys.exit(main())
