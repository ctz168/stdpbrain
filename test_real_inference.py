#!/usr/bin/env python3
"""
真实推理引擎测试脚本
使用真实的Qwen3.5-0.8B模型进行推理测试
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
print("类人脑双系统AI - 真实推理引擎测试")
print("=" * 70)

# 强制使用CPU，避免内存问题
device = 'cpu'
torch.set_num_threads(4)  # 限制线程数

# 1. 加载配置
print("\n[1] 加载配置...")
from configs.arch_config import BrainAIConfig
config = BrainAIConfig()
config.model_path = "/home/z/my-project/stdpbrain/models/Qwen3.5-0.8B"
config.quantization = "FP32"  # 使用FP32避免量化问题
print(f"   模型路径: {config.model_path}")
print(f"   量化方式: {config.quantization}")

# 2. 加载真实模型
print("\n[2] 加载真实Qwen模型...")
from core.qwen_interface import QwenInterface

model = QwenInterface(
    model_path=config.model_path,
    config=config,
    device=device,
    quantization='FP32'
)
print(f"   模型加载成功")
print(f"   参数量: {sum(p.numel() for p in model.model.parameters()):,}")

# 3. 初始化海马体系统
print("\n[3] 初始化海马体系统...")
from hippocampus.hippocampus_system import HippocampusSystem
hippocampus = HippocampusSystem(config, device=device)
print(f"   海马体初始化成功")

# 4. 初始化STDP引擎
print("\n[4] 初始化STDP引擎...")
from core.stdp_engine import STDPEngine
stdp_engine = STDPEngine(config, device=device)
print(f"   STDP引擎初始化成功")

# 5. 初始化自闭环优化器（连接真实模型）
print("\n[5] 初始化自闭环优化器...")
from self_loop.self_loop_optimizer import SelfLoopOptimizer
self_loop = SelfLoopOptimizer(config, model=model)
print(f"   自闭环优化器初始化成功")

# 测试结果记录
test_results = {
    'start_time': datetime.now().isoformat(),
    'config': {
        'model_path': config.model_path,
        'quantization': 'FP32',
        'device': device
    },
    'tests': []
}

def run_real_inference_test(test_id, input_text, expected_keywords=None):
    """运行真实推理测试"""
    print(f"\n{'='*70}")
    print(f"测试 #{test_id}: {input_text[:50]}...")
    print(f"{'='*70}")
    
    test_start = time.time()
    result = {
        'test_id': test_id,
        'input': input_text,
        'timestamp': datetime.now().isoformat(),
        'expected_keywords': expected_keywords or []
    }
    
    # 1. 真实模型推理
    print(f"\n[推理中...]")
    inference_start = time.time()
    try:
        output = model.generate(
            input_text,
            max_tokens=100,
            temperature=0.7,
            use_self_loop=False  # 直接推理，不使用自闭环
        )
        inference_time = time.time() - inference_start
        generated_text = output.text
        tokens_generated = len(output.tokens) if hasattr(output, 'tokens') else len(generated_text.split())
        
        print(f"   推理完成，耗时: {inference_time*1000:.1f}ms")
        print(f"   生成tokens: {tokens_generated}")
        print(f"   输出: {generated_text[:100]}...")
        
        result['inference'] = {
            'success': True,
            'output': generated_text,
            'tokens': tokens_generated,
            'time_ms': inference_time * 1000,
            'confidence': output.confidence if hasattr(output, 'confidence') else 0.8
        }
    except Exception as e:
        inference_time = time.time() - inference_start
        print(f"   推理失败: {e}")
        result['inference'] = {
            'success': False,
            'error': str(e),
            'time_ms': inference_time * 1000
        }
        return result
    
    # 2. 海马体记忆编码
    print(f"\n[海马体编码]")
    encode_start = time.time()
    try:
        # 使用模型隐藏状态作为特征
        # 简化：使用随机特征模拟（因为获取隐藏状态需要修改模型接口）
        features = torch.randn(1024)
        memory_id = hippocampus.encode(
            features=features,
            token_id=hash(input_text) % 100000,
            timestamp=int(time.time() * 1000),
            context=[{'content': input_text, 'semantic_pointer': input_text[:30]}]
        )
        encode_time = time.time() - encode_start
        print(f"   编码成功: {memory_id[:30]}...")
        print(f"   编码耗时: {encode_time*1000:.2f}ms")
        
        result['hippocampus_encode'] = {
            'success': True,
            'memory_id': memory_id,
            'time_ms': encode_time * 1000
        }
    except Exception as e:
        print(f"   编码失败: {e}")
        result['hippocampus_encode'] = {
            'success': False,
            'error': str(e)
        }
    
    # 3. 海马体记忆召回
    print(f"\n[海马体召回]")
    recall_start = time.time()
    try:
        recalled = hippocampus.recall(features, topk=3)
        recall_time = time.time() - recall_start
        print(f"   召回数量: {len(recalled)}")
        if recalled:
            print(f"   最相关记忆: {recalled[0]['semantic_pointer'][:50]}...")
        
        result['hippocampus_recall'] = {
            'success': True,
            'recall_count': len(recalled),
            'top_memory': recalled[0]['semantic_pointer'] if recalled else None,
            'time_ms': recall_time * 1000
        }
    except Exception as e:
        print(f"   召回失败: {e}")
        result['hippocampus_recall'] = {
            'success': False,
            'error': str(e)
        }
    
    # 4. STDP更新
    print(f"\n[STDP更新]")
    stdp_start = time.time()
    try:
        timestamp = time.time() * 1000
        stdp_engine.record_activation('attention', hash(input_text) % 1000, timestamp)
        stdp_engine.record_activation('ffn', test_id, timestamp)
        stdp_engine.set_contribution('attention', result['inference'].get('confidence', 0.5))
        
        # 执行STDP step
        stdp_engine.step(
            model_components={},
            inputs={'context_tokens': torch.tensor([1, 2, 3]), 'current_token': 4},
            outputs={'evaluation_score': 30 + test_id}
        )
        
        stdp_time = time.time() - stdp_start
        stats = stdp_engine.get_stats()
        print(f"   STDP周期: {stats['cycle_count']}")
        print(f"   更新耗时: {stdp_time*1000:.2f}ms")
        
        result['stdp_update'] = {
            'success': True,
            'cycle_count': stats['cycle_count'],
            'time_ms': stdp_time * 1000
        }
    except Exception as e:
        print(f"   STDP更新失败: {e}")
        result['stdp_update'] = {
            'success': False,
            'error': str(e)
        }
    
    # 5. 关键词匹配检查
    if expected_keywords:
        matched = []
        for kw in expected_keywords:
            if kw.lower() in generated_text.lower():
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
    
    total_time = time.time() - test_start
    result['total_time_ms'] = total_time * 1000
    print(f"\n[总耗时: {total_time*1000:.1f}ms]")
    
    # 清理内存
    gc.collect()
    
    return result

# 执行测试用例
print("\n" + "=" * 70)
print("开始执行真实推理测试")
print("=" * 70)

test_cases = [
    ("你好，请介绍一下你自己", ["AI", "模型", "助手"]),
    ("1+1等于多少？", ["2", "二"]),
    ("中国的首都是哪里？", ["北京"]),
    ("请写一句问候语", ["你好", "您好"]),
    ("什么是机器学习？", ["学习", "数据", "算法"]),
]

for i, (input_text, keywords) in enumerate(test_cases, 1):
    result = run_real_inference_test(i, input_text, keywords)
    test_results['tests'].append(result)
    
    # 每次测试后清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# 统计结果
print("\n" + "=" * 70)
print("测试统计")
print("=" * 70)

inference_success = sum(1 for t in test_results['tests'] if t.get('inference', {}).get('success', False))
encode_success = sum(1 for t in test_results['tests'] if t.get('hippocampus_encode', {}).get('success', False))
recall_success = sum(1 for t in test_results['tests'] if t.get('hippocampus_recall', {}).get('success', False))
stdp_success = sum(1 for t in test_results['tests'] if t.get('stdp_update', {}).get('success', False))

avg_inference_time = sum(
    t.get('inference', {}).get('time_ms', 0) 
    for t in test_results['tests'] 
    if t.get('inference', {}).get('success', False)
) / max(inference_success, 1)

total_tests = len(test_results['tests'])

print(f"\n推理成功率: {inference_success}/{total_tests} ({inference_success/total_tests*100:.0f}%)")
print(f"编码成功率: {encode_success}/{total_tests} ({encode_success/total_tests*100:.0f}%)")
print(f"召回成功率: {recall_success}/{total_tests} ({recall_success/total_tests*100:.0f}%)")
print(f"STDP成功率: {stdp_success}/{total_tests} ({stdp_success/total_tests*100:.0f}%)")
print(f"平均推理时间: {avg_inference_time:.1f}ms")

# 关键词匹配统计
keyword_matches = []
for t in test_results['tests']:
    if 'keyword_match' in t:
        keyword_matches.append(t['keyword_match']['match_rate'])
if keyword_matches:
    print(f"平均关键词匹配率: {sum(keyword_matches)/len(keyword_matches)*100:.0f}%")

# 海马体统计
hc_stats = hippocampus.get_stats()
print(f"\n海马体记忆数: {hc_stats['num_memories']}")
print(f"内存使用: {hc_stats['memory_usage_mb']:.4f}MB")

# STDP统计
stdp_stats = stdp_engine.get_stats()
print(f"STDP周期数: {stdp_stats['cycle_count']}")

# 保存结果
test_results['end_time'] = datetime.now().isoformat()
test_results['summary'] = {
    'total_tests': total_tests,
    'inference_success_rate': inference_success / total_tests,
    'encode_success_rate': encode_success / total_tests,
    'recall_success_rate': recall_success / total_tests,
    'stdp_success_rate': stdp_success / total_tests,
    'avg_inference_time_ms': avg_inference_time,
    'hippocampus_memories': hc_stats['num_memories'],
    'stdp_cycles': stdp_stats['cycle_count']
}

output_path = '/home/z/my-project/stdpbrain/real_test_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    # 转换不可序列化的数据
    serializable = json.loads(json.dumps(test_results, default=str, ensure_ascii=False))
    json.dump(serializable, f, ensure_ascii=False, indent=2)
print(f"\n测试结果已保存: {output_path}")

print("\n" + "=" * 70)
print("真实推理引擎测试完成")
print("=" * 70)
