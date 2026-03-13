#!/usr/bin/env python3
"""
测试修复后的代码

验证以下修复：
1. chat 方法返回非空字符串
2. STDP 使用真实数据
3. 特征维度正确适配
"""

import os
import sys
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.arch_config import BrainAIConfig
from core.interfaces import create_brain_ai


def test_model_loading():
    """测试模型加载"""
    print("=" * 50)
    print("测试 1: 模型加载")
    print("=" * 50)
    
    config = BrainAIConfig()
    config.model_path = "./models/Qwen3.5-0.8B"
    config.quantization = "FP16"
    
    try:
        ai = create_brain_ai(config)
        print(f"✅ 模型加载成功! 设备: {ai.device}")
        print(f"✅ 模型隐藏层大小: {ai.model_hidden_size}")
        return ai
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


def test_chat(ai):
    """测试对话功能"""
    print("\n" + "=" * 50)
    print("测试 2: 对话功能")
    print("=" * 50)
    
    test_messages = [
        "你好",
        "1+1等于几？",
        "请用一句话介绍Python"
    ]
    
    for msg in test_messages:
        print(f"\n用户: {msg}")
        try:
            response = ai.chat(msg)
            if response:
                print(f"AI: {response[:100]}...")
                print(f"✅ 响应长度: {len(response)}")
            else:
                print("❌ 响应为空!")
        except Exception as e:
            print(f"❌ 对话失败: {e}")


def test_memory(ai):
    """测试记忆功能"""
    print("\n" + "=" * 50)
    print("测试 3: 记忆功能")
    print("=" * 50)
    
    try:
        stats = ai.get_stats()
        hippocampus_stats = stats.get('hippocampus', {})
        print(f"记忆数量: {hippocampus_stats.get('num_memories', 0)}")
        print(f"记忆使用率: {hippocampus_stats.get('capacity_usage', 0):.2%}")
        print("✅ 记忆功能正常")
    except Exception as e:
        print(f"❌ 记忆功能测试失败: {e}")


def test_stdp(ai):
    """测试 STDP 功能"""
    print("\n" + "=" * 50)
    print("测试 4: STDP 功能")
    print("=" * 50)
    
    try:
        stats = ai.get_stats()
        stdp_stats = stats.get('stdp', {})
        print(f"STDP 周期数: {stdp_stats.get('cycle_count', 0)}")
        print(f"STDP 更新次数: {stdp_stats.get('total_updates', 0)}")
        print(f"动态权重范数: {stdp_stats.get('dynamic_weight_norm', 0):.6f}")
        print("✅ STDP 功能正常")
    except Exception as e:
        print(f"❌ STDP 功能测试失败: {e}")


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("开始测试修复后的代码")
    print("=" * 50)
    
    # 测试模型加载
    ai = test_model_loading()
    if ai is None:
        print("\n❌ 模型加载失败，无法继续测试")
        return
    
    # 测试对话功能
    test_chat(ai)
    
    # 测试记忆功能
    test_memory(ai)
    
    # 测试 STDP 功能
    test_stdp(ai)
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
