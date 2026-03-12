#!/usr/bin/env python3
"""
完整集成测试 - 测试BrainAI接口
验证海马体记忆与模型推理的深度集成
"""

import sys
import os
sys.path.insert(0, '/home/z/my-project/stdpbrain')

import torch
import time
import json
import gc
from datetime import datetime

print("=" * 70)
print("类人脑双系统AI - 完整集成测试")
print("=" * 70)

device = 'cpu'
torch.set_num_threads(4)

# 1. 加载完整BrainAI接口
print("\n[1] 加载完整BrainAI接口...")
from configs.arch_config import BrainAIConfig
from core.interfaces import BrainAIInterface

config = BrainAIConfig()
config.model_path = "/home/z/my-project/stdpbrain/models/Qwen3.5-0.8B"
config.quantization = "FP32"

# 创建BrainAI实例
ai = BrainAIInterface(config, device=device)
print("   BrainAI接口加载成功")

# 获取初始状态
initial_stats = ai.get_stats()
print(f"   初始海马体记忆数: {initial_stats['hippocampus']['num_memories']}")
print(f"   初始STDP周期: {initial_stats['stdp']['cycle_count']}")

# 测试结果
test_results = {
    'start_time': datetime.now().isoformat(),
    'tests': []
}

def run_integrated_test(test_id, input_text, expected_keywords=None):
    """运行集成测试"""
    print(f"\n{'='*70}")
    print(f"集成测试 #{test_id}: {input_text[:50]}...")
    print(f"{'='*70}")
    
    test_start = time.time()
    result = {
        'test_id': test_id,
        'input': input_text,
        'timestamp': datetime.now().isoformat(),
        'expected_keywords': expected_keywords or []
    }
    
    # 1. 获取测试前状态
    stats_before = ai.get_stats()
    result['state_before'] = {
        'hippocampus_memories': stats_before['hippocampus']['num_memories'],
        'stdp_cycles': stats_before['stdp']['cycle_count'],
        'dynamic_weight_norm': stats_before['stdp']['dynamic_weight_norm']
    }
    
    # 2. 执行对话（使用真实推理）
    print(f"\n[执行对话]")
    chat_start = time.time()
    try:
        response = ai.chat(
            user_input=input_text,
            history=None,
            max_tokens=80,
            thinking=True  # 启用思考模式
        )
        chat_time = time.time() - chat_start
        
        print(f"   响应: {response[:150]}...")
        print(f"   耗时: {chat_time*1000:.1f}ms")
        
        result['chat'] = {
            'success': True,
            'response': response,
            'time_ms': chat_time * 1000
        }
    except Exception as e:
        chat_time = time.time() - chat_start
        print(f"   对话失败: {e}")
        result['chat'] = {
            'success': False,
            'error': str(e),
            'time_ms': chat_time * 1000
        }
        return result
    
    # 3. 获取测试后状态
    stats_after = ai.get_stats()
    result['state_after'] = {
        'hippocampus_memories': stats_after['hippocampus']['num_memories'],
        'stdp_cycles': stats_after['stdp']['cycle_count'],
        'dynamic_weight_norm': stats_after['stdp']['dynamic_weight_norm']
    }
    
    # 4. 分析变化
    memory_growth = stats_after['hippocampus']['num_memories'] - stats_before['hippocampus']['num_memories']
    stdp_growth = stats_after['stdp']['cycle_count'] - stats_before['stdp']['cycle_count']
    weight_change = stats_after['stdp']['dynamic_weight_norm'] - stats_before['stdp']['dynamic_weight_norm']
    
    print(f"\n[状态变化]")
    print(f"   海马体记忆增长: {memory_growth}")
    print(f"   STDP周期增长: {stdp_growth}")
    print(f"   动态权重变化: {weight_change:.6f}")
    
    result['changes'] = {
        'memory_growth': memory_growth,
        'stdp_growth': stdp_growth,
        'weight_change': weight_change
    }
    
    # 5. 关键词匹配
    if expected_keywords:
        matched = []
        for kw in expected_keywords:
            if kw.lower() in response.lower():
                matched.append(kw)
        result['keyword_match'] = {
            'expected': expected_keywords,
            'matched': matched,
            'match_rate': len(matched) / len(expected_keywords) if expected_keywords else 0
        }
        print(f"\n[关键词匹配]")
        print(f"   期望: {expected_keywords}")
        print(f"   匹配: {matched}")
        print(f"   匹配率: {result['keyword_match']['match_rate']*100:.0f}%")
    
    # 6. 测试记忆召回
    print(f"\n[记忆召回测试]")
    try:
        recent_memories = ai._recall_recent_memories(topk=3)
        print(f"   召回记忆数: {len(recent_memories)}")
        if recent_memories:
            for i, mem in enumerate(recent_memories[:2]):
                print(f"   记忆{i+1}: {mem['semantic_pointer'][:40]}...")
        
        result['memory_recall'] = {
            'success': True,
            'recall_count': len(recent_memories),
            'memories': [m['semantic_pointer'] for m in recent_memories]
        }
    except Exception as e:
        print(f"   召回失败: {e}")
        result['memory_recall'] = {
            'success': False,
            'error': str(e)
        }
    
    total_time = time.time() - test_start
    result['total_time_ms'] = total_time * 1000
    print(f"\n[总耗时: {total_time*1000:.1f}ms]")
    
    gc.collect()
    return result

# 执行测试
print("\n" + "=" * 70)
print("开始执行集成测试")
print("=" * 70)

test_cases = [
    ("你好，我是张三，来自北京", ["张三", "北京"]),
    ("请记住我的名字是张三", ["张三"]),
    ("我叫什么名字？", ["张三"]),
    ("我来自哪个城市？", ["北京"]),
    ("请总结一下我们刚才的对话", ["张三", "北京"]),
]

for i, (input_text, keywords) in enumerate(test_cases, 1):
    result = run_integrated_test(i, input_text, keywords)
    test_results['tests'].append(result)
    gc.collect()

# 统计结果
print("\n" + "=" * 70)
print("集成测试统计")
print("=" * 70)

chat_success = sum(1 for t in test_results['tests'] if t.get('chat', {}).get('success', False))
recall_success = sum(1 for t in test_results['tests'] if t.get('memory_recall', {}).get('success', False))
total_memory_growth = sum(t.get('changes', {}).get('memory_growth', 0) for t in test_results['tests'])
total_stdp_growth = sum(t.get('changes', {}).get('stdp_growth', 0) for t in test_results['tests'])

avg_chat_time = sum(
    t.get('chat', {}).get('time_ms', 0) 
    for t in test_results['tests'] 
    if t.get('chat', {}).get('success', False)
) / max(chat_success, 1)

total_tests = len(test_results['tests'])

print(f"\n对话成功率: {chat_success}/{total_tests} ({chat_success/total_tests*100:.0f}%)")
print(f"记忆召回成功率: {recall_success}/{total_tests} ({recall_success/total_tests*100:.0f}%)")
print(f"总记忆增长: {total_memory_growth}")
print(f"总STDP周期增长: {total_stdp_growth}")
print(f"平均对话时间: {avg_chat_time:.1f}ms")

# 关键词匹配统计
keyword_matches = []
for t in test_results['tests']:
    if 'keyword_match' in t:
        keyword_matches.append(t['keyword_match']['match_rate'])
if keyword_matches:
    print(f"平均关键词匹配率: {sum(keyword_matches)/len(keyword_matches)*100:.0f}%")

# 最终状态
final_stats = ai.get_stats()
print(f"\n最终海马体记忆数: {final_stats['hippocampus']['num_memories']}")
print(f"最终STDP周期: {final_stats['stdp']['cycle_count']}")
print(f"最终动态权重范数: {final_stats['stdp']['dynamic_weight_norm']:.6f}")

# 保存结果
test_results['end_time'] = datetime.now().isoformat()
test_results['summary'] = {
    'total_tests': total_tests,
    'chat_success_rate': chat_success / total_tests,
    'recall_success_rate': recall_success / total_tests,
    'total_memory_growth': total_memory_growth,
    'total_stdp_growth': total_stdp_growth,
    'avg_chat_time_ms': avg_chat_time,
    'final_hippocampus_memories': final_stats['hippocampus']['num_memories'],
    'final_stdp_cycles': final_stats['stdp']['cycle_count']
}

output_path = '/home/z/my-project/stdpbrain/integration_test_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    serializable = json.loads(json.dumps(test_results, default=str, ensure_ascii=False))
    json.dump(serializable, f, ensure_ascii=False, indent=2)
print(f"\n测试结果已保存: {output_path}")

print("\n" + "=" * 70)
print("完整集成测试完成")
print("=" * 70)
