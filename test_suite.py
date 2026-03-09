#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全模块集成测试套件"""

import sys
import os
sys.path.insert(0, '/Users/hilbert/Desktop/stdpbrian')

print("=" * 60)
print("类人脑 AI 架构 - 全模块集成测试")
print("=" * 60)

def test_imports():
    """测试 1: 导入所有模块"""
   print("\n[测试 1] 模块导入测试...")
   print("-" * 60)
    
    modules = [
        'configs.arch_config',
        'hippocampus.hippocampus_system',
        'core.stdp_engine',
        'core.refresh_engine',
        'core.interfaces_working',
        'self_loop.self_loop_optimizer',
        'self_loop.self_evaluation',
        'self_loop.self_game',
        'evaluation.edge_performance_eval',
        'evaluation.hippocampus_eval',
        'evaluation.base_capability_eval',
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
           print(f"  OK: {module}")
        except Exception as e:
           print(f"  FAILED: {module} - {e}")
            failed.append(module)
    
   print(f"\n导入结果：{len(modules) - len(failed)}/{len(modules)} 成功")
   return len(failed) == 0


def test_hippocampus():
    """测试 2: 海马体系统"""
   print("\n[测试 2] 海马体系统测试...")
   print("-" * 60)
    
    try:
        import torch
       from configs.arch_config import default_config
       from hippocampus.hippocampus_system import HippocampusSystem
        
        config = default_config
        hippo = HippocampusSystem(config, device='cpu')
        
        features = torch.randn(1, 768)
        hippo.encode(features, token_id=100, timestamp=int(time.time()*1000), context=[])
        anchors = hippo.recall(features, topk=2)
        stats = hippo.get_stats()
        
       print(f"  编码：OK")
       print(f"  召回：{len(anchors)} 个锚点")
       print(f"  统计：{stats}")
       print("海马体测试通过!")
       return True
    except Exception as e:
       print(f"海马体测试失败：{e}")
        import traceback
        traceback.print_exc()
       return False


def test_stdp():
    """测试 3: STDP 引擎"""
   print("\n[测试 3] STDP 引擎测试...")
   print("-" * 60)
    
    try:
        import torch
       from configs.arch_config import default_config
       from core.stdp_engine import STDPEngine
        
        config = default_config
        stdp = STDPEngine(config, device='cpu')
        
        mock_inputs = {'context_tokens': [1, 2], 'current_token': 3, 'features': torch.randn(1, 768)}
        mock_outputs = {'attention_output': torch.randn(1, 768), 'ffn_output': torch.randn(1, 768)}
        stdp.step({'attention': None, 'ffn': None}, mock_inputs, mock_outputs, timestamp=time.time()*1000)
        stats = stdp.get_stats()
        
       print(f"  Step 执行：OK")
       print(f"  统计：{stats}")
       print("STDP 测试通过!")
       return True
    except Exception as e:
       print(f"STDP 测试失败：{e}")
        import traceback
        traceback.print_exc()
       return False


def test_self_loop():
    """测试 4: 自闭环优化器"""
   print("\n[测试 4] 自闭环优化器测试...")
   print("-" * 60)
    
    try:
       from configs.arch_config import default_config
       from self_loop.self_loop_optimizer import SelfLoopOptimizer
       from core.interfaces_working import SimpleLanguageModel
        
        config = default_config
        model = SimpleLanguageModel()
        optimizer = SelfLoopOptimizer(config, model=model)
        
       result1 = optimizer.run("你好", mode=1)
       print(f"  自组合模式：{result1.output_text[:30]}...")
        
       result2 = optimizer.run("什么是 AI?", mode=2)
       print(f"  自博弈模式：{result2.output_text[:30]}...")
        
       result3 = optimizer.run("如何学习？", mode=3)
       print(f"  自评估模式：{result3.output_text[:30]}...")
        
        stats = optimizer.get_stats()
       print(f"  统计：{stats}")
       print("自闭环测试通过!")
       return True
    except Exception as e:
       print(f"自闭环测试失败：{e}")
        import traceback
        traceback.print_exc()
       return False


def test_brain_ai():
    """测试 5: BrainAI 接口"""
   print("\n[测试 5] BrainAI 接口测试...")
   print("-" * 60)
    
    try:
       from core.interfaces_working import create_brain_ai
        
        ai = create_brain_ai(device='cpu')
       print(f"  实例创建：OK, 模型={type(ai.model).__name__}")
        
        output = ai.generate("你好", max_tokens=50)
       print(f"  文本生成：{output.text[:30]}...")
        
       response = ai.chat("介绍一下你自己")
       print(f"  对话接口：{response[:30]}...")
        
        stats = ai.get_stats()
       print(f"  统计：系统={stats['system']['model_type']}")
       print("BrainAI 测试通过!")
       return True
    except Exception as e:
       print(f"BrainAI 测试失败：{e}")
        import traceback
        traceback.print_exc()
       return False


def test_refresh_engine():
    """测试 6: 刷新引擎"""
   print("\n[测试 6] 100Hz 刷新引擎测试...")
   print("-" * 60)
    
    try:
        import time
       from configs.arch_config import default_config
       from hippocampus.hippocampus_system import HippocampusSystem
       from core.stdp_engine import STDPEngine
       from core.refresh_engine import RefreshCycleEngine
       from core.interfaces_working import SimpleLanguageModel
        
        config = default_config
        model = SimpleLanguageModel()
        hippo = HippocampusSystem(config, device='cpu')
        stdp = STDPEngine(config, device='cpu')
        
        engine = RefreshCycleEngine(
            model=model, 
            hippocampus=hippo, 
            stdp_engine=stdp, 
            period_ms=10, 
            narrow_window_size=2, 
           device='cpu'
        )
        
        for i in range(3):
           result = engine.run_cycle(input_token=i, input_text=f"Token {i}")
           print(f"  周期{i+1}: 耗时{result.cycle_time_ms:.2f}ms, 成功={result.success}")
        
        stats = engine.get_stats()
       print(f"  统计：平均周期={stats['avg_cycle_time_ms']:.2f}ms")
       print("刷新引擎测试通过!")
       return True
    except Exception as e:
       print(f"刷新引擎测试失败：{e}")
        import traceback
        traceback.print_exc()
       return False


def test_evaluators():
    """测试 7: 评估器"""
   print("\n[测试 7] 评估器测试...")
   print("-" * 60)
    
    try:
       from evaluation.edge_performance_eval import EdgePerformanceEvaluator
       from evaluation.hippocampus_eval import HippocampusEvaluator
       from evaluation.base_capability_eval import BaseCapabilityEvaluator
        
        edge = EdgePerformanceEvaluator(model_path=None)
        edge_score = edge.evaluate()
       print(f"  端侧性能：{edge_score:.3f}")
        
        hippo = HippocampusEvaluator(model_path=None)
        hippo_score = hippo.evaluate()
       print(f"  海马体能力：{hippo_score:.3f}")
        
        base = BaseCapabilityEvaluator(model_path=None)
        base_score = base.evaluate()
       print(f"  基础能力：{base_score:.3f}")
        
       print("评估器测试通过!")
       return True
    except Exception as e:
       print(f"评估器测试失败：{e}")
        import traceback
        traceback.print_exc()
       return False


def main():
    """运行所有测试"""
    import time
    
   results = {}
    
   results['imports'] = test_imports()
    if not results['imports']:
       print("\n导入测试失败，跳过后续测试")
       return False
    
   results['hippocampus'] = test_hippocampus()
   results['stdp'] = test_stdp()
   results['self_loop'] = test_self_loop()
   results['brain_ai'] = test_brain_ai()
   results['refresh_engine'] = test_refresh_engine()
   results['evaluators'] = test_evaluators()
    
   print("\n" + "=" * 60)
   print("测试结果汇总")
   print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "通过" if result else "失败"
       print(f"  {test_name}: {status}")
    
   print(f"\n总计：{passed}/{total} 通过 ({passed/total*100:.1f}%)")
   print("=" * 60)
    
   return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
