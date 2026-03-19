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
        print(f"User: {user_input}")
        
        start_time = time.time()
        
        # We use chat instead of chat_stream for clean output capture
        response = ai.chat(user_input, history)
        
        elapsed = time.time() - start_time
        print(f"AI: {response}")
        print(f"[Time: {elapsed:.2f}s]")
        
        # Check monologue
        if hasattr(ai, 'monologue_history') and ai.monologue_history:
            print(f"Inner Thought: {ai.monologue_history[-1]}")
            
        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        
        # Small delay to simulate thinking/processing gap if needed
        await asyncio.sleep(1)

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(run_capability_test())
