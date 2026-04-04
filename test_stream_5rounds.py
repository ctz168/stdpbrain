#!/usr/bin/env python3
"""STDPBrain 流式聊天测试 - 5轮独白思维流评估"""
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

print("="*60)
print("STDPBrain 流式聊天测试 - 5轮独白思维流评估")
print("="*60)

# 5轮连续对话
questions = [
    "你好，我想了解你的思考方式。",
    "你能跟我聊聊你内心是怎么想的吗？",
    "告诉我一个你最近想到的有趣的事情。",
    "你觉得人类意识和AI意识有什么区别？",
    "最后，你觉得你自己是一个什么样的存在？",
]

results = []
history = []
t_total = time.time()

for i, question in enumerate(questions):
    print(f"\n{'='*60}")
    print(f"第{i+1}轮对话")
    print(f"用户: {question}")
    print(f"{'='*60}")
    
    t1 = time.time()
    
    try:
        # 使用 chat 方法（thinking=False 但记录输出）
        response = ai.chat(question, history=history[-8:], max_tokens=120, thinking=False)
        elapsed = time.time() - t1
    except Exception as e:
        response = f"[错误] {e}"
        elapsed = time.time() - t1
    
    # 检查乱码
    garbage_words = ['painsWatch','becks','Progr','ilde','watch','andering','disposto','Bomb']
    gc = sum(1 for w in garbage_words if w in response)
    
    # 评估思维连贯性
    is_coherent = gc == 0 and len(response) > 20 and any('\u4e00' <= c <= '\u9fff' for c in response)
    
    print(f"AI: {response[:200]}...")
    print(f"[耗时: {elapsed:.1f}s, 乱码词:{gc}, 连贯: {'✅' if is_coherent else '❌'}]")
    
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})
    results.append({
        "round": i+1, "question": question, "response": response.strip(),
        "time": elapsed, "garbage_count": gc, "is_coherent": is_coherent
    })

total_time = time.time() - t_total

# 保存结果
os.makedirs('/home/z/my-project/download', exist_ok=True)
with open('/home/z/my-project/download/stream_test_5rounds.txt', 'w', encoding='utf-8') as f:
    f.write("STDPBrain 流式聊天测试 - 5轮独白思维流评估\n")
    f.write(f"总耗时: {total_time:.1f}s\n{'='*50}\n\n")
    for r in results:
        f.write(f"[第{r['round']}轮] ({r['time']:.1f}s) 连贯:{'✅' if r['is_coherent'] else '❌'} 乱码:{r['garbage_count']}\n")
        f.write(f"  Q: {r['question']}\n")
        f.write(f"  A: {r['response']}\n\n")
    
    coherent_count = sum(1 for r in results if r['is_coherent'])
    total_gc = sum(r['garbage_count'] for r in results)
    f.write(f"\n连贯轮数: {coherent_count}/5\n")
    f.write(f"总乱码词: {total_gc}\n")

print(f"\n{'='*60}")
print(f"=== 5轮评估 ===")
print(f"连贯轮数: {sum(1 for r in results if r['is_coherent'])}/5")
print(f"总乱码词: {sum(r['garbage_count'] for r in results)}")
print(f"总耗时: {total_time:.1f}s")
