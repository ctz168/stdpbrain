#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/hilbert/Desktop/stdpbrian')

print('=' * 60)
print('Functional Test Suite')
print('=' * 60)

# Test 1: Hippocampus System
print('\n[Test 1] Hippocampus System...')
try:
  import torch
  import time
  from configs.arch_config import default_config
  from hippocampus.hippocampus_system import HippocampusSystem
  
  config = default_config
  hippo = HippocampusSystem(config, device='cpu')
  
  features = torch.randn(1, 768)
  hippo.encode(features, token_id=100, timestamp=int(time.time()*1000), context=[])
  anchors = hippo.recall(features, topk=2)
  stats = hippo.get_stats()
  
  print(f'  Encode: OK')
  print(f'  Recall: {len(anchors)} anchors')
  print(f'  Stats: {stats}')
  print('Hippocampus: PASSED')
except Exception as e:
  print(f'Hippocampus: FAILED - {e}')

# Test 2: STDP Engine
print('\n[Test 2] STDP Engine...')
try:
  from core.stdp_engine import STDPEngine
  
  config = default_config
  stdp = STDPEngine(config, device='cpu')
  
  mock_inputs = {'context_tokens': [1, 2], 'current_token': 3, 'features': torch.randn(1, 768)}
  mock_outputs = {'attention_output': torch.randn(1, 768), 'ffn_output': torch.randn(1, 768)}
  stdp.step({'attention': None, 'ffn': None}, mock_inputs, mock_outputs, timestamp=time.time()*1000)
  stats = stdp.get_stats()
  
  print(f'  Step: OK')
  print(f'  Stats: {stats}')
  print('STDP: PASSED')
except Exception as e:
  print(f'STDP: FAILED - {e}')

# Test 3: Self-Loop Optimizer
print('\n[Test 3] Self-Loop Optimizer...')
try:
  from self_loop.self_loop_optimizer import SelfLoopOptimizer
  from core.interfaces_working import SimpleLanguageModel
  
  model = SimpleLanguageModel()
  optimizer = SelfLoopOptimizer(config, model=model)
  
  result1 = optimizer.run("你好")
  print(f'  Default mode: {result1.output_text[:30]}...')
  
  result2 = optimizer.run("什么是 AI?")
  print(f'  Another run: {result2.output_text[:30]}...')
  
  stats = optimizer.get_stats()
  print(f'  Stats: {stats}')
  print('Self-Loop: PASSED')
except Exception as e:
  print(f'Self-Loop: FAILED - {e}')

# Test 4: BrainAI Interface
print('\n[Test 4] BrainAI Interface...')
try:
  from core.interfaces_working import create_brain_ai
  
  ai = create_brain_ai(device='cpu')
  print(f'  Create: OK, model={type(ai.model).__name__}')
  
  output = ai.generate("你好", max_tokens=50)
  print(f'  Generate: {output.text[:30]}...')
  
  response = ai.chat("介绍一下你自己")
  print(f'  Chat: {response[:30]}...')
  
  stats = ai.get_stats()
  print(f'  Stats: system={stats.get("system", {}).get("model_type", "N/A")}')
  print('BrainAI: PASSED')
except Exception as e:
  print(f'BrainAI: FAILED - {e}')

# Test 5: Refresh Engine
print('\n[Test 5] 100Hz Refresh Engine...')
try:
  import time
  from core.refresh_engine import RefreshCycleEngine
  
  model = SimpleLanguageModel()
  hippo = HippocampusSystem(config, device='cpu')
  stdp = STDPEngine(config, device='cpu')
  
  engine = RefreshCycleEngine(model=model, hippocampus=hippo, stdp_engine=stdp, period_ms=10, narrow_window_size=2, device='cpu')
  
  for i in range(3):
   result = engine.run_cycle(input_token=i, input_text=f"Token {i}")
   print(f'  Cycle {i+1}: {result.cycle_time_ms:.2f}ms, success={result.success}')
  
  stats = engine.get_stats()
  print(f'  Stats: avg_cycle={stats["avg_cycle_time_ms"]:.2f}ms')
  print('Refresh Engine: PASSED')
except Exception as e:
  print(f'Refresh Engine: FAILED - {e}')

print('\n' + '=' * 60)
print('All functional tests complete!')
print('=' * 60)
