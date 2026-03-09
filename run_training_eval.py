#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行完整训练和评估流程"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, '/Users/hilbert/Desktop/stdpbrian')

def main():
   print("=" * 70)
   print("类人脑 AI 架构 - 训练和评估流程")
   print("=" * 70)
   print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   print("=" * 70)
    
   results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'scores': {}
    }
    
    # 1. 环境检查
   print("\n[1] 环境检查...")
   try:
       import torch
       from configs.arch_config import default_config
        
       os.makedirs('outputs/training', exist_ok=True)
       os.makedirs('outputs/evaluation', exist_ok=True)
        
       results['tests']['environment'] = {
            'status': 'passed',
            'python_version': sys.version.split()[0],
            'torch_version': torch.__version__,
            'device': 'cpu'
        }
       print(f"  Python: {sys.version.split()[0]}")
       print(f"  PyTorch: {torch.__version__}")
       print(f"  设备：cpu")
       print("  ✓ 通过")
    except Exception as e:
       results['tests']['environment'] = {'status': 'failed', 'error': str(e)}
       print(f"  ✗ 失败：{e}")
       return
    
    # 2. BrainAI 接口测试
   print("\n[2] BrainAI 接口测试...")
   try:
       from core.interfaces_working import create_brain_ai
        
       ai = create_brain_ai(device='cpu')
       response = ai.chat("你好，介绍一下你自己")
        
       results['tests']['brain_ai'] = {
            'status': 'passed',
            'response_length': len(response),
            'sample': response[:50]
        }
       print(f"  响应：{response[:50]}...")
       print("  ✓ 通过")
    except Exception as e:
       results['tests']['brain_ai'] = {'status': 'failed', 'error': str(e)}
       print(f"  ✗ 失败：{e}")
    
    # 3. 自闭环优化器测试
   print("\n[3] 自闭环优化器测试...")
   try:
       from self_loop.self_loop_optimizer import SelfLoopOptimizer
       from core.interfaces_working import SimpleLanguageModel
        
       model = SimpleLanguageModel()
        optimizer= SelfLoopOptimizer(config=default_config, model=model)
        
       result = optimizer.run("测试问题")
        
       results['tests']['self_loop'] = {
            'status': 'passed',
            'output': result.output_text[:30],
            'stats': optimizer.get_stats()
        }
       print(f"  输出：{result.output_text[:30]}...")
       print(f"  统计：cycle_count={optimizer.get_stats()['cycle_count']}")
       print("  ✓ 通过")
    except Exception as e:
       results['tests']['self_loop'] = {'status': 'failed', 'error': str(e)}
       print(f"  ✗ 失败：{e}")
    
    # 4. 刷新引擎测试
   print("\n[4] 100Hz 刷新引擎测试...")
   try:
       from core.refresh_engine import RefreshCycleEngine
       from hippocampus.hippocampus_system import HippocampusSystem
       from core.stdp_engine import STDPEngine
        
        engine = RefreshCycleEngine(
           model=model,
            hippocampus=HippocampusSystem(config=default_config, device='cpu'),
            stdp_engine=STDPEngine(config=default_config, device='cpu'),
            period_ms=10,
           narrow_window_size=2,
           device='cpu'
        )
        
        cycles = []
        for i in range(3):
           result = engine.run_cycle(input_token=i, input_text=f"Token {i}")
            cycles.append({
                'cycle_time_ms': result.cycle_time_ms,
                'success': result.success
            })
        
       results['tests']['refresh_engine'] = {
            'status': 'passed',
            'cycles': cycles,
            'avg_cycle_time_ms': sum(c['cycle_time_ms'] for c in cycles) / len(cycles)
        }
       print(f"  平均周期：{results['tests']['refresh_engine']['avg_cycle_time_ms']:.2f}ms")
       print("  ✓ 通过")
    except Exception as e:
       results['tests']['refresh_engine'] = {'status': 'failed', 'error': str(e)}
       print(f"  ✗ 失败：{e}")
    
    # 5. 模型评估
   print("\n[5] 模型评估...")
    
    # 5.1 基础能力
   print("  [5.1] 基础能力评估...")
   try:
       from evaluation.base_capability_eval import BaseCapabilityEvaluator
        
       evaluator = BaseCapabilityEvaluator(ai_interface=None)
       score = evaluator.evaluate()
        
       results['scores']['base_capability'] = score
       print(f"    得分：{score:.3f}")
    except Exception as e:
       results['scores']['base_capability'] = 0
       print(f"    ✗ 失败：{e}")
    
    # 5.2 推理能力
   print("  [5.2] 推理能力评估...")
   try:
       from evaluation.reasoning_eval import ReasoningEvaluator
        
       evaluator = ReasoningEvaluator(ai_interface=None)
       score = evaluator.evaluate()
        
       results['scores']['reasoning'] = score
       print(f"    得分：{score:.3f}")
    except Exception as e:
       results['scores']['reasoning'] = 0
       print(f"    ✗ 失败：{e}")
    
    # 5.3 海马体能力
   print("  [5.3] 海马体能力评估...")
   try:
       from evaluation.hippocampus_eval import HippocampusEvaluator
        
       evaluator = HippocampusEvaluator(ai_interface=None)
       score = evaluator.evaluate()
        
       results['scores']['hippocampus'] = score
       print(f"    得分：{score:.3f}")
    except Exception as e:
       results['scores']['hippocampus'] = 0
       print(f"    ✗ 失败：{e}")
    
    # 5.4 自闭环优化
   print("  [5.4] 自闭环优化评估...")
   try:
       from evaluation.self_loop_eval import SelfLoopEvaluator
        
       evaluator = SelfLoopEvaluator(ai_interface=None)
       score = evaluator.evaluate()
        
       results['scores']['self_loop'] = score
       print(f"    得分：{score:.3f}")
    except Exception as e:
       results['scores']['self_loop'] = 0
       print(f"    ✗ 失败：{e}")
    
    # 6. 计算综合评分
   print("\n[6] 综合评分...")
    valid_scores = [v for v in results['scores'].values() if v > 0]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
       results['overall_score'] = avg_score
       print(f"  综合评分：{avg_score:.3f} / 1.000")
    else:
       results['overall_score'] = 0
       print("  无有效评分")
    
    # 7. 保存结果
   print("\n[7] 保存结果...")
    
    # JSON 格式
   with open('outputs/training_eval_results.json', 'w', encoding='utf-8') as f:
       json.dump(results, f, ensure_ascii=False, indent=2)
   print("  outputs/training_eval_results.json")
    
    # 文本报告
   report = f"""
{'='*70}
类人脑 AI 架构 - 训练评估报告
{'='*70}
时间：{results['timestamp']}
环境：Python {results['tests'].get('environment', {}).get('python_version', 'N/A')}, 
      PyTorch {results['tests'].get('environment', {}).get('torch_version', 'N/A')}

测试结果:
"""
    
    for test_name, test_result in results['tests'].items():
        status = "✓" if test_result.get('status') == 'passed' else"✗"
       report += f"  {status} {test_name}\n"
    
   report += f"\n评估得分:\n"
    for score_name, score_value in results['scores'].items():
       report += f"  {score_name}: {score_value:.3f}\n"
    
   report += f"\n{'='*70}\n"
   report += f"综合评分：{results.get('overall_score', 0):.3f} / 1.000\n"
   report += f"{'='*70}\n"
    
   with open('outputs/training_eval_report.txt', 'w', encoding='utf-8') as f:
       f.write(report)
   print("  outputs/training_eval_report.txt")
    
    # 打印报告
   print("\n" + report)
    
   print("=" * 70)
   print("训练评估完成!")
   print("=" * 70)

if __name__ == "__main__":
    main()
