#!/usr/bin/env python3
"""快速评估脚本 - 测试核心功能"""
import sys
sys.path.insert(0, '/Users/hilbert/Desktop/stdpbrian')

print("Quick Evaluation Test")
print("=" * 50)

# Test imports
print("\n1. Imports...")
try:
  import torch
  from configs.arch_config import default_config
  print("  OK: torch + config")
except Exception as e:
  print(f"  FAIL: {e}")
  sys.exit(1)

# Test BrainAI
print("\n2. BrainAI Interface...")
try:
  from core.interfaces_working import create_brain_ai
    ai = create_brain_ai(device='cpu')
  response = ai.chat("你好")
  print(f"  OK: {response[:30]}...")
except Exception as e:
  print(f"  FAIL: {e}")

# Test Self-Loop
print("\n3. Self-Loop Optimizer...")
try:
  from self_loop.self_loop_optimizer import SelfLoopOptimizer
  from core.interfaces_working import SimpleLanguageModel
   model = SimpleLanguageModel()
    opt = SelfLoopOptimizer(config=default_config, model=model)
  result = opt.run("测试")
  print(f"  OK: {result.output_text[:30]}...")
except Exception as e:
  print(f"  FAIL: {e}")

# Test Evaluators
print("\n4. Evaluators...")
scores = {}
try:
  from evaluation.base_capability_eval import BaseCapabilityEvaluator
    base = BaseCapabilityEvaluator(ai_interface=None)
  scores['base'] = base.evaluate()
  print(f"  Base: {scores['base']:.3f}")
except Exception as e:
  print(f"  Base FAIL: {e}")

try:
  from evaluation.hippocampus_eval import HippocampusEvaluator
    he = HippocampusEvaluator(ai_interface=None)
  scores['hippo'] = he.evaluate()
  print(f"  Hippo: {scores['hippo']:.3f}")
except Exception as e:
  print(f"  Hippo FAIL: {e}")

# Summary
if scores:
   avg = sum(scores.values()) / len(scores)
  print(f"\nAverage Score: {avg:.3f}")
   
print("\n" + "=" * 50)
print("Test Complete!")
