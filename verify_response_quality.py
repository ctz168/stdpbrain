import torch
import sys
import os
import asyncio

# Add current directory to path
sys.path.append(os.getcwd())

from configs.arch_config import BrainAIConfig
from core.interfaces import BrainAIInterface

async def verify_response():
    print("Initializing AI...")
    config = BrainAIConfig()
    ai = BrainAIInterface(config)
    
    user_input = "你好呀，今天心情怎么样？"
    print(f"\nUser: {user_input}")
    
    print("\nGenerating response...")
    # Using chat instead of chat_stream for simpler verification
    response = ai.chat(user_input)
    
    print(f"\nAI: {response}")
    
    # Check monologue
    if hasattr(ai, 'monologue_history') and ai.monologue_history:
        print(f"\nLast Monologue: {ai.monologue_history[-1]}")

if __name__ == "__main__":
    asyncio.run(verify_response())
