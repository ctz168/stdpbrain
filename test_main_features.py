#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 main.py 的非交互式版本"""

import sys
sys.path.insert(0, '.')

from core.qwen_interface import create_qwen_ai

print("=" * 70)
print("类人脑 AI 架构 - 完整功能测试")
print("=" * 70)

ai = create_qwen_ai(model_path="./models/Qwen3.5-0.8B-Base", device="cpu")

test_cases = [
	 ("打招呼", "你好，请介绍一下自己"),
	 ("知识问答", "什么是人工智能？"),
	 ("创作", "写一首关于春天的短诗"),
]

print("\n开始测试...")
for name, prompt in test_cases:
	print(f"\n[{name}]")
	print(f"输入：{prompt}")
	response = ai.chat(prompt)
	print(f"输出：{response[:200]}...")

stats = ai.get_stats()
print("\n" + "=" * 70)
print("统计信息")
print("=" * 70)
print(f"生成次数：{stats['generation_count']}")
print(f"总 token 数：{stats['total_tokens']}")
print(f"设备：{stats['device']}")
if 'hippocampus' in stats:
	 hp = stats['hippocampus']
	print(f"海马体记忆数：{hp.get('num_memories', 0)}")
	print(f"内存使用：{hp.get('memory_usage_mb', 0):.2f} MB")

print("\n✅ 所有测试完成!")
