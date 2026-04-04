#!/usr/bin/env python3
"""快速聊天测试 - 精简版"""
import warnings
warnings.filterwarnings('ignore')
import sys, time, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

print('='*60)
print('STDPBrain 快速聊天测试 - Qwen3.5-0.8B')
print('='*60)

t0 = time.time()
import config as cfg
from configs.arch_config import BrainAIConfig
config = BrainAIConfig()
config.model_path = cfg.MODEL_PATH
config.quantization = cfg.QUANTIZATION

from core.interfaces import BrainAIInterface
ai = BrainAIInterface(config, device='cpu')
print(f'\n✅ 系统加载完成 ({time.time()-t0:.1f}s)')

# 测试问题
tests = [
    ("身份认知", "你好，用一句话介绍你自己。"),
    ("记忆注入", "我叫张三，我来自北京，是一名程序员。"),
    ("记忆召回", "你还记得我叫什么名字吗？"),
    ("推理测试", "如果A比B高，B比C高，那A和C谁更高？"),
]

results = []
history = []

for i, (name, question) in enumerate(tests):
    print(f'\n--- 第{i+1}轮: {name} ---')
    print(f'用户: {question}')
    
    t1 = time.time()
    try:
        response = ai.chat(question, history=history[-4:], max_tokens=80, thinking=False)
        elapsed = time.time() - t1
        print(f'AI: {response}')
        print(f'[耗时: {elapsed:.1f}s, 长度: {len(response)}字]')
        
        # 记录
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
        
        results.append({
            "round": i+1,
            "name": name,
            "question": question,
            "response": response.strip(),
            "time": elapsed,
            "is_garbled": len(set(response)) < 5 or response.count('of') > 3,
            "has_chinese": any('\u4e00' <= c <= '\u9fff' for c in response)
        })
    except Exception as e:
        print(f'❌ 错误: {e}')
        results.append({"round": i+1, "name": name, "error": str(e)})

# 评估报告
print('\n' + '='*60)
print('评估报告')
print('='*60)

for r in results:
    status = '❌ 乱码' if r.get('is_garbled', False) else ('✅ 正常' if r.get('has_chinese', False) else '⚠️ 疑似')
    print(f'\n[{status}] 第{r["round"]}轮 {r["name"]}:')
    print(f'  问题: {r["question"]}')
    if 'error' in r:
        print(f'  错误: {r["error"]}')
    else:
        print(f'  回复: {r["response"][:100]}...')
        print(f'  耗时: {r["time"]:.1f}s')

# 记忆和推理测试
print('\n' + '-'*40)
memory_test = any('张三' in r.get('response', '') for r in results if r.get('name') == '记忆召回')
reasoning_test = any('A' in r.get('response', '') for r in results if r.get('name') == '推理测试')

print(f'记忆召回测试: {"✅ 通过" if memory_test else "❌ 未通过"}')
print(f'推理能力测试: {"✅ 通过" if reasoning_test else "❌ 未通过"}')

total_time = time.time() - t0
print(f'\n总耗时: {total_time:.1f}s')

# 保存结果
os.makedirs('/home/z/my-project/download', exist_ok=True)
with open('/home/z/my-project/download/chat_test_round1.txt', 'w', encoding='utf-8') as f:
    f.write('STDPBrain 聊天测试报告 - Round 1\n')
    f.write('='*50 + '\n\n')
    for r in results:
        f.write(f'第{r["round"]}轮 [{r["name"]}]\n')
        f.write(f'问题: {r["question"]}\n')
        if 'error' in r:
            f.write(f'错误: {r["error"]}\n')
        else:
            f.write(f'回复: {r["response"]}\n')
            f.write(f'耗时: {r["time"]:.1f}s\n')
        f.write('\n')
    f.write(f'\n记忆召回: {"通过" if memory_test else "未通过"}\n')
    f.write(f'推理能力: {"通过" if reasoning_test else "未通过"}\n')
    f.write(f'总耗时: {total_time:.1f}s\n')

print(f'\n结果已保存到 /home/z/my-project/download/chat_test_round1.txt')
