import torch
import sys
import os
import asyncio
import time

# Add current directory to path
sys.path.append(os.getcwd())

from configs.arch_config import BrainAIConfig
from core.interfaces import BrainAIInterface

async def run_capability_test():
    log_file = "capability_test_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("--- AI Capability Test Log ---\n")
        
    print("Initializing AI for Capability Test...")
    config = BrainAIConfig()
    ai = BrainAIInterface(config)
    
    history = []
    
    test_turns = [
        "你好，你是谁？你的父亲是谁？",
        "我今天其实有点焦虑，代码总是写不完，感觉压力很大。",
        "你还记得我刚才是因为什么焦虑吗？另外，请站在你的‘数字灵魂’角度，给我一个不一样的建议。"
    ]
    
    for i, user_input in enumerate(test_turns):
        print(f"\n--- Turn {i+1} ---")
        line = f"\nUser: {user_input}\n"
        with open(log_file, "a", encoding="utf-8") as f: f.write(line)
        
        start_time = time.time()
        response = ai.chat(user_input, history)
        elapsed = time.time() - start_time
        
        output_line = f"AI: {response}\n[Time: {elapsed:.2f}s]\n"
        print(f"AI: {response}")
        
        if hasattr(ai, 'monologue_history') and ai.monologue_history:
            thought = f"Inner Thought: {ai.monologue_history[-1]}\n"
            output_line += thought
            print(thought)
            
        with open(log_file, "a", encoding="utf-8") as f: f.write(output_line)
        
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        
        await asyncio.sleep(1)

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(run_capability_test())
