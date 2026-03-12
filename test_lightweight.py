#!/usr/bin/env python3
"""
轻量级核心功能测试脚本
测试海马体系统、STDP引擎、自闭环优化器（不加载完整模型）
"""

import sys
import os
sys.path.insert(0, '/home/z/my-project/stdpbrain')

import torch
import time

print("=" * 60)
print("类人脑双系统AI - 轻量级核心功能测试")
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
    
    # 测试多次编码
    for i in range(5):
        test_features = torch.randn(1024)
        memory_id = hippocampus.encode(
            features=test_features,
            token_id=100 + i,
            timestamp=int(time.time() * 1000) + i,
            context=[{'content': f'测试记忆{i}', 'token_id': 100 + i}]
        )
    print(f"   ✓ 海马体编码成功 (5次)")
    
    # 测试召回
    recalled = hippocampus.recall(test_features, topk=3)
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
    test_cases = [
        ("你好，请介绍一下自己", "self_combine"),
        ("请帮我证明这道数学题", "self_game"),
        ("请给出一个重要的决策建议", "self_eval"),
        ("今天天气怎么样", "self_combine"),
        ("计算 123 * 456 等于多少", "self_game"),
    ]
    
    print(f"   ✓ 自闭环优化器初始化成功")
    for input_text, expected in test_cases:
        mode = self_loop.decide_mode(input_text)
        status = "✓" if mode == expected else "△"
        print(f"   {status} '{input_text[:20]}...' -> {mode}")
    
    # 测试运行
    result = self_loop.run("你好")
    print(f"   - 运行结果: {result.output_text[:50]}...")
    print(f"   - 使用模式: {result.mode_used}")
    print(f"   - 置信度: {result.confidence:.2f}")
    
except Exception as e:
    print(f"   ✗ 自闭环优化器测试失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试海马体子模块
print("\n[5] 测试海马体子模块...")
try:
    from hippocampus.ec_encoder import EntorhinalEncoder
    from hippocampus.dg_separator import DentateGyrusSeparator
    from hippocampus.ca3_memory import CA3EpisodicMemory
    from hippocampus.ca1_gate import CA1AttentionGate
    
    # EC编码器
    ec = EntorhinalEncoder(input_dim=1024, output_dim=64, sparsity=0.1)
    ec_output = ec.encode_single(torch.randn(1024))
    print(f"   ✓ EC编码器: 输出维度 {ec_output.shape}")
    
    # DG分离器
    dg = DentateGyrusSeparator(input_dim=64, output_dim=128, sparsity=0.1)
    dg_output, mem_id = dg.separate_and_id(ec_output)
    print(f"   ✓ DG分离器: 输出维度 {dg_output.shape}, ID: {mem_id[:20]}...")
    
    # CA3记忆
    ca3 = CA3EpisodicMemory(max_capacity=1000, feature_dim=128)
    ca3.store(mem_id, int(time.time() * 1000), "test_pointer", "", [], dg_output)
    recalled = ca3.complete_pattern({'features': dg_output}, topk=1)
    print(f"   ✓ CA3记忆: 存储/召回成功, 召回数: {len(recalled)}")
    
    # CA1门控
    ca1 = CA1AttentionGate(feature_dim=128, hidden_size=1024)
    query = torch.randn(1, 10, 1024)
    key = torch.randn(1, 10, 1024)
    gate = ca1(query, key, [{'semantic_pointer': 'test'}])
    print(f"   ✓ CA1门控: 输出形状 {gate.shape}")
    
except Exception as e:
    print(f"   ✗ 海马体子模块测试失败: {e}")
    import traceback
    traceback.print_exc()

# 6. 测试STDP规则
print("\n[6] 测试STDP规则...")
try:
    from core.stdp_engine import STDPRule
    import torch
    
    rule = STDPRule(alpha_LTP=0.01, beta_LTD=0.008, time_window_ms=20)
    
    # 测试LTP (Δt > 0)
    pre_times = torch.tensor([0.0, 5.0, 10.0])
    post_times = torch.tensor([10.0, 10.0, 10.0])
    contributions = torch.tensor([0.5, 0.5, 0.5])
    
    delta_w = rule.compute_update(pre_times, post_times, contributions)
    print(f"   ✓ STDP规则计算成功")
    print(f"   - LTP更新 (Δt>0): {delta_w.tolist()}")
    
    # 测试LTD (Δt < 0)
    pre_times2 = torch.tensor([15.0, 20.0, 25.0])
    post_times2 = torch.tensor([10.0, 10.0, 10.0])
    delta_w2 = rule.compute_update(pre_times2, post_times2, contributions)
    print(f"   - LTD更新 (Δt<0): {delta_w2.tolist()}")
    
except Exception as e:
    print(f"   ✗ STDP规则测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("轻量级核心功能测试完成")
print("=" * 60)
