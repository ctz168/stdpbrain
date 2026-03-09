#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""类人脑 AI 架构 - 训练优化测评流程"""

import sys, os, time, json
from datetime import datetime
sys.path.insert(0, '/Users/hilbert/Desktop/stdpbrian')

print("=" * 70)
print("训练、优化和测评全流程")
print("=" * 70)

print("\n[1] 准备...")
try:
  import torch
  from configs.arch_config import default_config
  os.makedirs('outputs/training', exist_ok=True)
  os.makedirs('outputs/evaluation', exist_ok=True)
  print("OK")
except Exception as e:
  print(f"FAIL: {e}")
  sys.exit(1)

print("\n[2] 预训练...")
try:
  from training.pretrain_adapter import PretrainAdapter
  from core.interfaces_working import SimpleLanguageModel
  model = SimpleLanguageModel()
   adapter= PretrainAdapter(model, config=default_config)
    train_data = {'samples': [{'input': '你好', 'output': '你好'}]}
  metrics = adapter.train(datasets=train_data, learning_rate=1e-5, batch_size=2, epochs=3)
  print(f"OK")
except Exception as e:
  print(f"FAIL: {e}")

print("\n[3] 在线学习...")
try:
  from core.stdp_engine import STDPEngine
  from hippocampus.hippocampus_system import HippocampusSystem
    stdp = STDPEngine(config=default_config, device='cpu')
    hippo = HippocampusSystem(config=default_config, device='cpu')
  print(f"OK")
except Exception as e:
  print(f"FAIL: {e}")

print("\n[4] 评估...")
scores = {}
try:
  from evaluation.base_capability_eval import BaseCapabilityEvaluator
    base = BaseCapabilityEvaluator(ai_interface=None)
  scores['base'] = base.evaluate()
  print(f"Base: {scores['base']:.3f}")
except Exception as e:
  print(f"FAIL: {e}")

try:
  from evaluation.hippocampus_eval import HippocampusEvaluator
    hippo_eval = HippocampusEvaluator(ai_interface=None)
  scores['hippocampus'] = hippo_eval.evaluate()
  print(f"Hippocampus: {scores['hippocampus']:.3f}")
except Exception as e:
  print(f"FAIL: {e}")

avg_score = sum(scores.values()) / len(scores) if scores else 0
print(f"\n综合评分：{avg_score:.3f}")
print("完成!")
