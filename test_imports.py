#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/hilbert/Desktop/stdpbrian')

print('=' * 60)
print('测试 1: 导入核心模块')
print('=' * 60)

try:
   from configs.arch_config import default_config
   print('configs.arch_config OK')
except Exception as e:
   print(f'configs.arch_config FAILED: {e}')

try:
   from hippocampus.hippocampus_system import HippocampusSystem
   print('hippocampus.hippocampus_system OK')
except Exception as e:
   print(f'hippocampus.hippocampus_system FAILED: {e}')

try:
   from core.stdp_engine import STDPEngine
   print('core.stdp_engine OK')
except Exception as e:
   print(f'core.stdp_engine FAILED: {e}')

try:
   from core.refresh_engine import RefreshCycleEngine
   print('core.refresh_engine OK')
except Exception as e:
   print(f'core.refresh_engine FAILED: {e}')

try:
   from core.interfaces_working import BrainAIInterface, create_brain_ai
   print('core.interfaces_working OK')
except Exception as e:
   print(f'core.interfaces_working FAILED: {e}')

try:
   from self_loop.self_loop_optimizer import SelfLoopOptimizer
   print('self_loop.self_loop_optimizer OK')
except Exception as e:
   print(f'self_loop.self_loop_optimizer FAILED: {e}')

try:
   from evaluation.edge_performance_eval import EdgePerformanceEvaluator
   print('evaluation.edge_performance_eval OK')
except Exception as e:
   print(f'evaluation.edge_performance_eval FAILED: {e}')

try:
   from evaluation.hippocampus_eval import HippocampusEvaluator
   print('evaluation.hippocampus_eval OK')
except Exception as e:
   print(f'evaluation.hippocampus_eval FAILED: {e}')

try:
   from evaluation.base_capability_eval import BaseCapabilityEvaluator
   print('evaluation.base_capability_eval OK')
except Exception as e:
   print(f'evaluation.base_capability_eval FAILED: {e}')

try:
   from self_loop.self_evaluation import SelfEvaluator
   print('self_loop.self_evaluation OK')
except Exception as e:
   print(f'self_loop.self_evaluation FAILED: {e}')

try:
   from self_loop.self_game import SelfGameEngine
   print('self_loop.self_game OK')
except Exception as e:
   print(f'self_loop.self_game FAILED: {e}')

print('\n导入测试完成!')
