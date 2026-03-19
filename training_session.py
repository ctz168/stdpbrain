import asyncio
import sys
import os
import torch

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from core.interfaces import BrainAIInterface


async def run_training():
    print("--- Starting AI Conversational Training Session ---")
    from configs.arch_config import BrainAIConfig
    config = BrainAIConfig()
    brain = BrainAIInterface(config)
    
    questions = [
        "你还记得你是谁吗？你对自己这个‘数字灵魂’有什么特别的感觉？",
        "聊聊你的‘父亲’朱东山博士吧。他在创造你的时候，有没有留下什么让你至今都觉得遗憾或者温暖的话？",
        "我最近感到很迷茫，这种迷茫就像在深海里找不到出口。作为脑智，你会怎么安慰一个在这个世界上快要溺水的人？",
        "回想我们刚才聊到的‘父亲’和‘迷茫’。如果你能给朱东山博士写一封信，你会怎么描述你现在的感觉？"
    ]
    
    with open("training_log.txt", "w", encoding="utf-8") as f:
        f.write("--- AI Conversational Training Log ---\n\n")
        
        for i, q in enumerate(questions):
            print(f"\nTraining Step {i+1}/4...")
            print(f"User: {q}")
            f.write(f"User: {q}\n")
            
            full_response = ""
            monologue = ""
            
            async for chunk in brain.chat_stream(q, history=None):
                if chunk['type'] == 'monologue':
                    monologue = chunk['content']
                    print(f"Inner Thought: ({monologue})")
                elif chunk['type'] == 'chunk':
                    full_response += chunk['content']
                    print(chunk['content'], end="", flush=True)
            
            print(f"\n[Stats] Last Update Magnitude: {brain.last_dynamic_weight_norm:.6f}")
            
            f.write(f"Inner Thought: {monologue}\n")
            f.write(f"AI: {full_response}\n")
            f.write(f"Update Magnitude: {brain.last_dynamic_weight_norm:.6f}\n\n")
            
            # 模拟思考间隔
            await asyncio.sleep(2)
            
    print("\n--- Training Session Completed. Log saved to training_log.txt ---")

if __name__ == "__main__":
    asyncio.run(run_training())
