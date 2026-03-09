#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试完整的 Qwen AI 系统"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("类人脑双系统全闭环 AI 架构 - 完整测试")
print("=" * 70)

# 测试 1: 导入模块
print("\n[测试 1] 导入模块...")
try:
    from core.qwen_interface import QwenBrainAI, create_qwen_ai
    print("✓ 模块导入成功")
except Exception as e:
    print(f"✗ 模块导入失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 2: 创建 AI 实例
print("\n[测试 2] 创建 AI 实例...")
try:
    ai = create_qwen_ai(
        model_path="./models/Qwen3.5-0.8B-Base",
        device="cpu",
        use_int4=False
    )
    print("✓ AI 实例创建成功")
except Exception as e:
    print(f"✗ AI 实例创建失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 3: 文本生成
print("\n[测试 3] 文本生成测试...")
test_prompts = [
    "你好，",
    "人工智能的核心是",
]

for prompt in test_prompts:
    print(f"\n输入：{prompt}")
    try:
        response = ai.generate(prompt, max_new_tokens=50)
        print(f"输出：{response[:100]}...")
    except Exception as e:
        print(f"✗ 生成失败：{e}")

# 测试 4: 对话测试
print("\n[测试 4] 对话测试...")
messages = [
    "你好，请介绍一下自己",
    "什么是类人脑 AI 架构？",
]

for msg in messages:
    print(f"\n你：{msg}")
    try:
        response = ai.chat(msg)
        print(f"AI: {response[:150]}...")
    except Exception as e:
        print(f"✗ 对话失败：{e}")

# 测试 5: 统计信息
print("\n[测试 5] 统计信息...")
stats = ai.get_stats()
print(f"生成次数：{stats.get('generation_count', 0)}")
print(f"总 token 数：{stats.get('total_tokens', 0)}")
print(f"设备：{stats.get('device', 'unknown')}")

if 'hippocampus' in stats:
    print("海马体系统：已加载")
else:
    print("海马体系统：未加载")

print("\n" + "=" * 70)
print("✅ 所有测试完成!")
print("=" * 70)
