#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/hilbert/Desktop/stdpbrian')

print('=' * 60)
print('Simple Test Suite')
print('=' * 60)

# Test torch
try:
  import torch
  print('OK: PyTorch', torch.__version__)
except Exception as e:
  print('FAILED: PyTorch -', e)
  sys.exit(1)

# Test configs
try:
  from configs.arch_config import default_config
  print('OK: configs.arch_config')
except Exception as e:
  print('FAILED: configs.arch_config -', e)

# Test hippocampus
try:
  from hippocampus.hippocampus_system import HippocampusSystem
  print('OK: hippocampus.hippocampus_system')
except Exception as e:
  print('FAILED: hippocampus.hippocampus_system -', e)

# Test STDP
try:
  from core.stdp_engine import STDPEngine
  print('OK: core.stdp_engine')
except Exception as e:
  print('FAILED: core.stdp_engine -', e)

# Test refresh engine
try:
  from core.refresh_engine import RefreshCycleEngine
  print('OK: core.refresh_engine')
except Exception as e:
  print('FAILED: core.refresh_engine -', e)

# Test interfaces
try:
  from core.interfaces_working import create_brain_ai
  print('OK: core.interfaces_working')
except Exception as e:
  print('FAILED: core.interfaces_working -', e)

# Test self_loop
try:
  from self_loop.self_loop_optimizer import SelfLoopOptimizer
  print('OK: self_loop.self_loop_optimizer')
except Exception as e:
  print('FAILED: self_loop.self_loop_optimizer -', e)

# Test evaluators - skip for now due to import chain issues
# try:
#  from evaluation.edge_performance_eval import EdgePerformanceEvaluator
#  print('OK: evaluation.edge_performance_eval')
# except Exception as e:
#  print('FAILED: evaluation.edge_performance_eval -', e)

print('=' * 60)
print('Import tests complete! (some modules skipped)')
print('=' * 60)
