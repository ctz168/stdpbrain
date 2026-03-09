#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试 Qwen 接口"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.qwen_interface import create_qwen_ai

print("=" * 60)
print("快速测试 Qwen 接口")
print("=" * 60)

# 创建 AI 实例
print("\n正在初始化 AI...")
ai = create_qwen_ai(
    model_path="./models/Qwen3.5-0.8B-Base",
    device="cpu",
    use_int4=False
)

# 测试对话
test_messages = [
    "你好",
    "类人脑架构是什么？",
]

for msg in test_messages:
    print(f"\n你：{msg}")
    response = ai.chat(msg)
    print(f"AI: {response[:150]}...")

# 统计
stats = ai.get_stats()
print(f"\n统计信息:")
print(f"  - 生成次数：{stats['generation_count']}")
print(f"  - 总 token 数：{stats['total_tokens']}")
print(f"  - 设备：{stats['device']}")

print("\n✅ 测试完成!")
