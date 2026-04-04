"""
STDPBrain 3轮聊天测试 - 专注内心独白流 + 正式回复
观察 inner monologue 和 speech output 的质量与流畅度
"""
import sys, os
sys.path.insert(0, '/home/z/my-project/stdpbrain')

from configs.arch_config import BrainAIConfig
import config as secret_config

config = BrainAIConfig()
config.model_path = secret_config.MODEL_PATH
config.quantization = getattr(secret_config, 'QUANTIZATION', config.quantization)

from core.interfaces import BrainAIInterface
ai = BrainAIInterface(config, device=None)

questions = [
    "你好，你喜欢什么？",
    "如果我告诉你今天下雨了，你会怎么想？",  
    "请帮我想一个创意短故事的开头"
]

history = []
results = []

for i, q in enumerate(questions):
    print(f"\n{'='*60}")
    print(f"第{i+1}轮")
    print(f"{'='*60}")
    print(f"用户: {q}")
    
    # Phase 1: Inner thought
    print("\n💭 [内心独白]", flush=True)
    thinking = ''
    state_before = ai.inner_thought_engine.mind_state.value
    for char in ai.inner_thought_engine.generate_inner_thought(external_stimulus=q, max_tokens=100):
        thinking += char
        print(char, end='', flush=True)
    state_after = ai.inner_thought_engine.mind_state.value
    urge = ai.inner_thought_engine._last_urge_to_speak
    print(f"\n  [状态:{state_before}→{state_after} 欲望:{urge:.2f} 字数:{len(thinking)}]")
    
    # Phase 2: Response
    print("\n🗣️ [正式回复]", flush=True)
    response = ai.chat(q, history=history[-6:], max_tokens=200, thinking=False)
    print(response)
    print(f"  [字数:{len(response)}]")
    
    history.append({'role': 'user', 'content': q})
    history.append({'role': 'assistant', 'content': response})
    
    results.append({
        'round': i+1,
        'question': q,
        'thinking': thinking,
        'thinking_len': len(thinking),
        'state_before': state_before,
        'state_after': state_after,
        'urge': urge,
        'response': response,
        'response_len': len(response)
    })

# Save report
report = "STDPBrain 3轮聊天测试报告\n"
report += "=" * 50 + "\n\n"
for r in results:
    report += f"第{r['round']}轮:\n"
    report += f"  问题: {r['question']}\n"
    report += f"  内心独白[{r['thinking_len']}字]: {r['thinking'][:300]}\n"
    report += f"  状态: {r['state_before']} → {r['state_after']}\n"
    report += f"  欲望: {r['urge']:.2f}\n"
    report += f"  正式回复[{r['response_len']}字]: {r['response'][:300]}\n\n"

os.makedirs('/home/z/my-project/download', exist_ok=True)
with open('/home/z/my-project/download/chat_3rounds_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print(f"\n报告已保存: /home/z/my-project/download/chat_3rounds_report.txt")
