#!/usr/bin/env python3
"""
类人脑AI - 10轮测试与优化循环
测试STDP学习、海马体记忆、自闭环优化
"""

import sys
import os
sys.path.insert(0, '/home/z/my-project/stdpbrain')

import torch
import time
import json
from datetime import datetime

print("=" * 70)
print("类人脑双系统AI - 10轮测试与优化循环")
print("=" * 70)

# 导入模块
from configs.arch_config import BrainAIConfig
from hippocampus.hippocampus_system import HippocampusSystem
from core.stdp_engine import STDPEngine
from self_loop.self_loop_optimizer import SelfLoopOptimizer

# 初始化配置
config = BrainAIConfig()
config.model_path = "/home/z/my-project/stdpbrain/models/Qwen3.5-0.8B"

# 初始化组件
hippocampus = HippocampusSystem(config, device='cpu')
stdp_engine = STDPEngine(config, device='cpu')
self_loop = SelfLoopOptimizer(config, model=None)

# 测试记录
test_results = []

def run_test_round(round_num, test_input, expected_keywords=None):
    """运行单轮测试"""
    print(f"\n{'='*70}")
    print(f"第 {round_num} 轮测试")
    print(f"{'='*70}")
    
    round_start = time.time()
    round_result = {
        'round': round_num,
        'input': test_input,
        'timestamp': datetime.now().isoformat(),
        'hippocampus_before': {},
        'hippocampus_after': {},
        'stdp_before': {},
        'stdp_after': {},
        'self_loop_result': None,
        'memory_recall_test': None,
        'improvements': []
    }
    
    # 1. 记录初始状态
    round_result['hippocampus_before'] = hippocampus.get_stats()
    round_result['stdp_before'] = stdp_engine.get_stats()
    
    print(f"\n[初始状态]")
    print(f"  海马体记忆数: {round_result['hippocampus_before']['num_memories']}")
    print(f"  STDP周期: {round_result['stdp_before']['cycle_count']}")
    
    # 2. 执行自闭环优化
    print(f"\n[执行自闭环优化]")
    result = self_loop.run(test_input)
    round_result['self_loop_result'] = {
        'output': result.output_text,
        'mode': result.mode_used,
        'confidence': result.confidence,
        'reasoning_trace': result.reasoning_trace[:3] if result.reasoning_trace else []
    }
    print(f"  模式: {result.mode_used}")
    print(f"  输出: {result.output_text[:80]}...")
    print(f"  置信度: {result.confidence:.2f}")
    
    # 3. 测试海马体记忆编码
    print(f"\n[海马体记忆编码]")
    test_features = torch.randn(1024)
    memory_id = hippocampus.encode(
        features=test_features,
        token_id=hash(test_input) % 100000,
        timestamp=int(time.time() * 1000),
        context=[{'content': test_input, 'semantic_pointer': test_input[:30]}]
    )
    print(f"  编码成功: {memory_id[:30]}...")
    
    # 4. 测试记忆召回
    print(f"\n[记忆召回测试]")
    recalled = hippocampus.recall(test_features, topk=3)
    round_result['memory_recall_test'] = {
        'recall_count': len(recalled),
        'top_memory': recalled[0]['semantic_pointer'] if recalled else None
    }
    print(f"  召回数量: {len(recalled)}")
    if recalled:
        print(f"  最相关记忆: {recalled[0]['semantic_pointer'][:50]}...")
    
    # 5. 记录STDP更新
    print(f"\n[STDP更新]")
    timestamp = time.time() * 1000
    stdp_engine.record_activation('attention', hash(test_input) % 1000, timestamp)
    stdp_engine.record_activation('ffn', round_num, timestamp)
    stdp_engine.set_contribution('attention', result.confidence)
    
    # 6. 记录最终状态
    round_result['hippocampus_after'] = hippocampus.get_stats()
    round_result['stdp_after'] = stdp_engine.get_stats()
    
    print(f"\n[最终状态]")
    print(f"  海马体记忆数: {round_result['hippocampus_after']['num_memories']}")
    print(f"  STDP周期: {round_result['stdp_after']['cycle_count']}")
    
    # 7. 分析改进
    mem_growth = round_result['hippocampus_after']['num_memories'] - round_result['hippocampus_before']['num_memories']
    if mem_growth > 0:
        round_result['improvements'].append(f"海马体记忆增长: +{mem_growth}")
    
    if result.confidence > 0.7:
        round_result['improvements'].append(f"高置信度响应: {result.confidence:.2f}")
    
    if len(recalled) > 0:
        round_result['improvements'].append(f"成功召回记忆: {len(recalled)}条")
    
    round_time = time.time() - round_start
    round_result['time_ms'] = round_time * 1000
    print(f"\n[本轮耗时: {round_time*1000:.1f}ms]")
    
    return round_result

# 定义10轮测试用例
test_cases = [
    ("你好，请介绍一下你自己", ["AI", "模型", "助手"]),
    ("我叫张三，来自北京，请记住我的名字", ["张三", "北京"]),
    ("你还记得我叫什么名字吗？", ["张三"]),
    ("请计算 15 + 27 等于多少", ["42"]),
    ("如果今天是星期三，后天是星期几？", ["星期五", "周五"]),
    ("请给我讲一个简短的故事", ["故事"]),
    ("我刚才告诉你我叫什么名字？", ["张三"]),
    ("请解释什么是机器学习", ["机器学习", "学习", "数据"]),
    ("我来自哪个城市？", ["北京"]),
    ("总结一下我们今天的对话内容", ["张三", "北京", "对话"]),
]

# 运行10轮测试
for i, (test_input, keywords) in enumerate(test_cases, 1):
    result = run_test_round(i, test_input, keywords)
    test_results.append(result)
    time.sleep(0.5)  # 短暂暂停

# 生成总结报告
print("\n" + "=" * 70)
print("测试总结报告")
print("=" * 70)

total_memories = hippocampus.get_stats()['num_memories']
total_stdp_cycles = stdp_engine.get_stats()['cycle_count']
avg_confidence = sum(r['self_loop_result']['confidence'] for r in test_results) / len(test_results)
avg_time = sum(r['time_ms'] for r in test_results) / len(test_results)

print(f"\n[整体统计]")
print(f"  总测试轮数: {len(test_results)}")
print(f"  最终海马体记忆数: {total_memories}")
print(f"  STDP总周期: {total_stdp_cycles}")
print(f"  平均置信度: {avg_confidence:.2f}")
print(f"  平均响应时间: {avg_time:.1f}ms")

print(f"\n[模式使用统计]")
mode_counts = {}
for r in test_results:
    mode = r['self_loop_result']['mode']
    mode_counts[mode] = mode_counts.get(mode, 0) + 1
for mode, count in mode_counts.items():
    print(f"  {mode}: {count}次")

print(f"\n[记忆召回成功率]")
recall_success = sum(1 for r in test_results if r['memory_recall_test']['recall_count'] > 0)
print(f"  成功召回: {recall_success}/{len(test_results)}次")

print(f"\n[改进记录]")
for r in test_results:
    if r['improvements']:
        print(f"  第{r['round']}轮: {', '.join(r['improvements'])}")

# 保存测试结果
report_path = "/home/z/my-project/stdpbrain/test_results.json"
with open(report_path, 'w', encoding='utf-8') as f:
    # 转换不可序列化的数据
    serializable_results = []
    for r in test_results:
        sr = {
            'round': r['round'],
            'input': r['input'],
            'timestamp': r['timestamp'],
            'self_loop_result': r['self_loop_result'],
            'memory_recall_test': r['memory_recall_test'],
            'improvements': r['improvements'],
            'time_ms': r['time_ms'],
            'hippocampus_memories_before': r['hippocampus_before']['num_memories'],
            'hippocampus_memories_after': r['hippocampus_after']['num_memories'],
        }
        serializable_results.append(sr)
    json.dump(serializable_results, f, ensure_ascii=False, indent=2)
print(f"\n测试结果已保存到: {report_path}")

print("\n" + "=" * 70)
print("10轮测试与优化循环完成")
print("=" * 70)
