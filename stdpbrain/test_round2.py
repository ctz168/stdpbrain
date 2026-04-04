#!/usr/bin/env python3
"""STDPBrain 聊天测试 - Round 2"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import config as cfg
from configs.arch_config import BrainAIConfig
config = BrainAIConfig()
config.model_path = cfg.MODEL_PATH
config.quantization = cfg.QUANTIZATION

from core.interfaces import BrainAIInterface
ai = BrainAIInterface(config, device='cpu')

tests = [
    ("身份认知", "你好，请介绍一下你自己。"),
    ("记忆注入", "我叫张三，我来自北京，今年28岁，是一名程序员。"),
    ("无关话题", "今天天气怎么样？"),
    ("记忆召回", "你还记得我的名字和职业吗？"),
    ("推理测试", "如果A比B高，B比C高，那A和C谁更高？"),
]

results = []
history = []
t_total = time.time()

for i, (name, question) in enumerate(tests):
    t1 = time.time()
    response = ai.chat(question, history=history[-8:], max_tokens=100, thinking=False)
    elapsed = time.time() - t1
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})
    results.append({"round": i+1, "name": name, "question": question, "response": response.strip(), "time": elapsed})
    print(f"[第{i+1}轮:{name}] {elapsed:.1f}s -> {response.strip()[:80]}...")

total_time = time.time() - t_total

memory_ok = any('张三' in r['response'] for r in results if r['name'] == '记忆召回')
reasoning_ok = any('A' in r['response'] and ('更高' in r['response'] or '高' in r['response']) for r in results if r['name'] == '推理测试')

os.makedirs('/home/z/my-project/download', exist_ok=True)
with open('/home/z/my-project/download/chat_test_round2.txt', 'w', encoding='utf-8') as f:
    f.write(f"STDPBrain 聊天测试 - Round 2 (STDP重新启用, lr=0.001)\n总耗时: {total_time:.1f}s\n{'='*50}\n\n")
    for r in results:
        f.write(f"[第{r['round']}轮] {r['name']} ({r['time']:.1f}s)\n")
        f.write(f"  Q: {r['question']}\n")
        f.write(f"  A: {r['response']}\n\n")
    f.write(f"\n记忆召回: {'✅通过' if memory_ok else '❌未通过'}\n")
    f.write(f"推理能力: {'✅通过' if reasoning_ok else '❌未通过'}\n")

print(f"\n=== 评估 ===")
print(f"记忆召回: {'✅通过' if memory_ok else '❌未通过'}")
print(f"推理能力: {'✅通过' if reasoning_ok else '❌未通过'}")
print(f"总耗时: {total_time:.1f}s")
