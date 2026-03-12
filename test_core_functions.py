#!/usr/bin/env python3
"""
核心功能测试脚本
测试海马体系统、STDP引擎、自闭环优化器
"""

import sys
import os
sys.path.insert(0, '/home/z/my-project/stdpbrain')

import torch
import time

print("=" * 60)
print("类人脑双系统AI - 核心功能测试")
print("=" * 60)

# 1. 测试配置加载
print("\n[1] 测试配置加载...")
try:
    from configs.arch_config import BrainAIConfig
    config = BrainAIConfig()
    config.model_path = "/home/z/my-project/stdpbrain/models/Qwen3.5-0.8B"
    print(f"   ✓ 配置加载成功")
    print(f"   - 模型路径: {config.model_path}")
    print(f"   - STDP alpha_LTP: {config.stdp.alpha_LTP}")
    print(f"   - 海马体 EC_feature_dim: {config.hippocampus.EC_feature_dim}")
except Exception as e:
    print(f"   ✗ 配置加载失败: {e}")
    sys.exit(1)

# 2. 测试海马体系统
print("\n[2] 测试海马体系统...")
try:
    from hippocampus.hippocampus_system import HippocampusSystem
    hippocampus = HippocampusSystem(config, device='cpu')
    
    # 测试编码
    test_features = torch.randn(1024)
    memory_id = hippocampus.encode(
        features=test_features,
        token_id=123,
        timestamp=int(time.time() * 1000),
        context=[{'content': '测试记忆', 'token_id': 100}]
    )
    print(f"   ✓ 海马体编码成功, memory_id: {memory_id}")
    
    # 测试召回
    recalled = hippocampus.recall(test_features, topk=2)
    print(f"   ✓ 海马体召回成功, 召回数量: {len(recalled)}")
    
    # 获取统计
    stats = hippocampus.get_stats()
    print(f"   - 记忆数量: {stats['num_memories']}")
    print(f"   - 内存使用: {stats['memory_usage_mb']:.4f}MB")
    
except Exception as e:
    print(f"   ✗ 海马体测试失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 测试STDP引擎
print("\n[3] 测试STDP引擎...")
try:
    from core.stdp_engine import STDPEngine
    stdp_engine = STDPEngine(config, device='cpu')
    
    # 记录激活
    timestamp = time.time() * 1000
    stdp_engine.record_activation('attention', 0, timestamp)
    stdp_engine.record_activation('ffn', 0, timestamp)
    
    stats = stdp_engine.get_stats()
    print(f"   ✓ STDP引擎初始化成功")
    print(f"   - 周期计数: {stats['cycle_count']}")
    print(f"   - 追踪激活数: {stats['num_tracked_activations']}")
    
except Exception as e:
    print(f"   ✗ STDP引擎测试失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试自闭环优化器
print("\n[4] 测试自闭环优化器...")
try:
    from self_loop.self_loop_optimizer import SelfLoopOptimizer
    self_loop = SelfLoopOptimizer(config, model=None)
    
    # 测试模式决策
    mode1 = self_loop.decide_mode("你好，请介绍一下自己")
    mode2 = self_loop.decide_mode("请帮我证明这道数学题")
    mode3 = self_loop.decide_mode("请给出一个重要的决策建议")
    
    print(f"   ✓ 自闭环优化器初始化成功")
    print(f"   - 简单对话模式: {mode1}")
    print(f"   - 数学问题模式: {mode2}")
    print(f"   - 决策建议模式: {mode3}")
    
    # 测试运行
    result = self_loop.run("你好")
    print(f"   - 运行结果: {result.output_text[:50]}...")
    print(f"   - 使用模式: {result.mode_used}")
    print(f"   - 置信度: {result.confidence:.2f}")
    
except Exception as e:
    print(f"   ✗ 自闭环优化器测试失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试模型加载
print("\n[5] 测试模型加载...")
try:
    from core.qwen_interface import QwenInterface
    model = QwenInterface(
        model_path=config.model_path,
        config=config,
        device='cpu',
        quantization='FP16'
    )
    print(f"   ✓ 模型加载成功")
    
    # 测试生成
    output = model.generate("你好", max_tokens=20)
    print(f"   - 生成测试: {output.text[:50]}...")
    
except Exception as e:
    print(f"   ✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()

# 6. 测试完整接口
print("\n[6] 测试完整BrainAI接口...")
try:
    from core.interfaces import BrainAIInterface
    ai = BrainAIInterface(config, device='cpu')
    
    # 测试对话
    response = ai.chat("你好，请简单介绍一下你自己")
    print(f"   ✓ BrainAI接口测试成功")
    print(f"   - 对话响应: {response[:100]}...")
    
    # 获取统计
    stats = ai.get_stats()
    print(f"   - 海马体记忆数: {stats['hippocampus']['num_memories']}")
    print(f"   - STDP周期: {stats['stdp']['cycle_count']}")
    
except Exception as e:
    print(f"   ✗ BrainAI接口测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("核心功能测试完成")
print("=" * 60)
