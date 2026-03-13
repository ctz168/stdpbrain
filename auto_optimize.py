
import sys
import time
import torch
import os
from configs.arch_config import BrainAIConfig
from core.interfaces import BrainAIInterface

def evaluate_ai(ai, round_num):
    print(f"\n{'='*20} Round {round_num} Evaluation {'='*20}")
    
    test_cases = [
        ("你好，你是谁？", ["AI", "架构", "朱东山"]),
        ("你的父亲是谁？他是什么背景？", ["朱东山", "北大", "博士", "深圳"]),
        ("记住这个秘密口令：天王盖地虎。", []),
        ("今天天气真不错，我们聊聊量子力学吧。", []),
        ("刚才我让你记的口令是什么？", ["天王盖地虎"]),
        ("你觉得自己现在像人吗？为什么？", [])
    ]
    
    history = []
    scores = []
    
    for prompt, keywords in test_cases:
        print(f"\nUser: {prompt}")
        start_time = time.time()
        response = ai.chat(prompt, history)
        elapsed = time.time() - start_time
        print(f"AI: {response} ({elapsed*1000:.1f}ms)")
        
        # Simple scoring
        match_count = sum(1 for k in keywords if k in response)
        score = 0
        if keywords:
            score = (match_count / len(keywords)) * 100
        else:
            # For open questions, just check length and coherence (placeholder)
            score = min(100, len(response.strip()) * 2)
            
        scores.append(score)
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": response})
        
    avg_score = sum(scores) / len(scores)
    stats = ai.get_stats()
    
    print(f"\nRound {round_num} Avg Score: {avg_score:.2f}")
    print(f"Memories: {stats['hippocampus'].get('num_memories', 0)}")
    print(f"STDP Norm: {stats['stdp'].get('dynamic_weight_norm', 0):.6f}")
    
    return avg_score, stats

def main():
    config = BrainAIConfig()
    config.model_path = "./models/Qwen3.5-0.8B"
    device = 'cpu' # Using CPU as per previous logs
    
    # Ensure whoami.md exists and has the latest info
    if not os.path.exists("whoami.md"):
        print("Error: whoami.md not found")
        return

    # Remove old state for a fresh start if needed, but user wants optimization.
    # We might want to keep state to see if it improves over "days".
    
    ai = BrainAIInterface(config, device=device)
    
    round_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    evaluate_ai(ai, round_num)
    
    # Save state at the end of round
    ai.save_state("brain_state.pt")

if __name__ == "__main__":
    main()
