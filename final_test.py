#!/usr/bin/env python
"""测试 Qwen AI 完整功能"""
import sys
sys.path.insert(0, '.')
from core.qwen_interface import create_qwen_ai

print("类人脑 AI 架构 - 完整功能测试")
print("=" * 50)

ai = create_qwen_ai(model_path="./models/Qwen3.5-0.8B-Base", device="cpu")

test_cases = [
    ("打招呼", "你好"),
    ("知识问答", "什么是人工智能？"),
]

for name, prompt in test_cases:
   print(f"\n[{name}] 输入：{prompt}")
   response = ai.chat(prompt)
   print(f"输出：{response[:150]}...")

stats = ai.get_stats()
print(f"\n统计：生成{stats['generation_count']}次，{stats['total_tokens']} tokens")
print("测试完成!")
