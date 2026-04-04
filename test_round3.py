#!/usr/bin/env python3
"""STDPBrain 聊天测试 - Round 3 (深入测试)"""
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
    ("自我意识", "你有自己的意识和思考能力吗？"),
    ("创意写作", "用三句话写一个关于月亮的微型故事。"),
    ("逻辑推理", "一个农夫有17只羊，除了9只以外都死了，农夫还剩几只羊？"),
    ("情感理解", "如果一个人说'我没事'但语气很悲伤，你觉得他真正想表达什么？"),
    ("代码能力", "用Python写一个判断一个数是否为质数的函数。"),
]

results = []
history = []
t_total = time.time()

for i, (name, question) in enumerate(tests):
    t1 = time.time()
    response = ai.chat(question, history=history[-8:], max_tokens=120, thinking=False)
    elapsed = time.time() - t1
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})
    results.append({"round": i+1, "name": name, "question": question, "response": response.strip(), "time": elapsed})
    print(f"[第{i+1}轮:{name}] {elapsed:.1f}s -> {response.strip()[:100]}...")

total_time = time.time() - t_total

os.makedirs('/home/z/my-project/download', exist_ok=True)
with open('/home/z/my-project/download/chat_test_round3.txt', 'w', encoding='utf-8') as f:
    f.write(f"STDPBrain 聊天测试 - Round 3 (深入测试)\n总耗时: {total_time:.1f}s\n{'='*50}\n\n")
    for r in results:
        f.write(f"[第{r['round']}轮] {r['name']} ({r['time']:.1f}s)\n")
        f.write(f"  Q: {r['question']}\n")
        f.write(f"  A: {r['response']}\n\n")

garbage_words = ['painsWatch','becks','Progr','ilde','watch','andering','disposto']
total_text = ' '.join(r['response'] for r in results)
garbage_count = sum(1 for w in garbage_words if w in total_text)
f.write(f"\n乱码词统计: {garbage_count}\n")

print(f"\n=== 汇总 ===")
print(f"乱码词统计: {garbage_count}")
print(f"总耗时: {total_time:.1f}s")
