#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""环境检查脚本 - 验证所有依赖是否就绪"""

import sys
import subprocess

def print_header(text):
  print("\n" + "=" * 60)
  print(text.center(60))
  print("=" * 60)

def check_python_version():
   """检查 Python 版本"""
  print_header("Python 版本检查")
   version = sys.version.split()[0]
  print(f"当前 Python 版本：{version}")
   
   # 检查是否为 3.11
   if version.startswith("3.11"):
      print("✓ Python 版本正确 (3.11.x)")
      return True
   else:
      print(f"✗ Python 版本不匹配 (需要 3.11.x, 当前 {version})")
      print("  建议：conda create -n stdpbrain python=3.11")
      return False

def check_torch():
   """检查 PyTorch 安装"""
  print_header("PyTorch 检查")
    try:
      import torch
      print(f"PyTorch 版本：{torch.__version__}")
      print(f"CUDA 可用：{torch.cuda.is_available()}")
        if torch.cuda.is_available():
          print(f"CUDA 版本：{torch.version.cuda}")
      print("✓ PyTorch 已安装")
       return True
  except ImportError:
      print("✗ PyTorch 未安装")
      print("  安装命令：conda install pytorch cpuonly -c pytorch")
      return False

def check_transformers():
   """检查 Transformers 库"""
  print_header("Transformers 检查")
    try:
      import transformers
      print(f"Transformers 版本：{transformers.__version__}")
      print("✓ Transformers 已安装")
      return True
  except ImportError:
      print("✗ Transformers 未安装")
      print("  安装命令：pip install transformers sentencepiece accelerate optimum")
      return False

def check_other_deps():
   """检查其他依赖"""
  print_header("其他依赖检查")
    
  deps = {
        'numpy': 'numpy',
       'scipy': 'scipy',
       'sklearn': 'scikit-learn',
       'pandas': 'pandas',
       'yaml': 'pyyaml',
       'tqdm': 'tqdm'
    }
    
   all_ok = True
    for module_name, pip_name in deps.items():
       try:
           __import__(module_name)
           module = sys.modules[module_name]
           version = getattr(module, '__version__', 'unknown')
          print(f"✓ {pip_name}: {version}")
      except ImportError:
          print(f"✗ {pip_name} 未安装")
           all_ok = False
    
    if not all_ok:
      print("\n  安装命令：pip install numpy scipy scikit-learn pandas tqdm pyyaml")
    
   return all_ok

def check_model_files():
   """检查模型文件"""
  print_header("模型文件检查")
  import os
    
   model_paths = [
        './models/Qwen3.5-0.8B-Base',
       './models',
       './checkpoints'
    ]
    
    for path in model_paths:
       if os.path.exists(path):
          print(f"✓ 模型目录存在：{path}")
          return True
    
  print("✗ 未找到模型目录")
  print("  请确保模型文件位于 ./models/Qwen3.5-0.8B-Base")
   return False

def check_project_structure():
   """检查项目结构"""
  print_header("项目结构检查")
  import os
    
  required_dirs = [
        'core',
       'hippocampus',
       'self_loop',
       'training',
       'evaluation',
       'configs',
       'outputs'
    ]
    
   all_ok = True
    for dir_name in required_dirs:
       if os.path.exists(dir_name):
          print(f"✓ 目录存在：{dir_name}")
       else:
          print(f"✗ 目录缺失：{dir_name}")
           all_ok = False
    
   return all_ok

def main():
  print("\n" + "🧠" * 30)
  print("类人脑双系统 AI 架构 - 环境检查工具".center(60))
  print("🧠" * 30)
  print(f"\n检查时间：{subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}")
    
  results = {}
    
   # 执行所有检查
  results['python'] = check_python_version()
  results['torch'] = check_torch()
  results['transformers'] = check_transformers()
  results['other_deps'] = check_other_deps()
  results['model'] = check_model_files()
  results['structure'] = check_project_structure()
    
   # 汇总结果
  print_header("检查汇总")
    
   passed = sum(results.values())
   total = len(results)
    
    for check_name, passed_check in results.items():
      status = "✓" if passed_check else"✗"
      print(f"{status} {check_name}")
    
  print(f"\n总计：{passed}/{total} 项检查通过")
    
    if passed == total:
      print("\n✅ 所有环境检查通过！可以运行训练和评估")
      print("\n运行命令:")
      print("  python run_training_eval.py     # 完整训练评估")
      print("  python functional_test.py       # 功能测试")
      print("  python quick_eval.py            # 快速评估")
   else:
      print("\n❌ 部分检查未通过，请先安装缺失的依赖")
      print("\n快速安装命令:")
      print(" conda create-n stdpbrain python=3.11")
      print(" conda activate stdpbrain")
      print(" conda install pytorch cpuonly -c pytorch")
      print("  pip install transformers sentencepiece accelerate optimum")
      print("  pip install numpy scipy scikit-learn pandas tqdm pyyaml")
    
  print("\n" + "=" * 60)

if __name__ == "__main__":
  main()
