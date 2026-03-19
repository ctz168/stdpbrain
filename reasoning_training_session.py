import asyncio
import torch
import logging
import sys
import os
from core.interfaces import BrainAIInterface
from configs.arch_config import BrainAIConfig

# 设置日志
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

async def run_reasoning_training():
    print("\n--- Starting AI Reasoning Training Suite ---\n")
    
    config = BrainAIConfig()
    config.model_path = "./models/Qwen3.5-0.8B"
    config.stdp.enabled = True
    
    # 初始化大脑
    brain = BrainAIInterface(config)
    
    # 推理训练题目
    reasoning_tasks = [
        {
            "user": "逻辑测试1：如果所有的 A 都是 B，而所有的 B 都是 C。那么，所有的 A 必然是 C 吗？请解释你的推理过程。",
            "focus": "演绎推理与传递律"
        },
        {
            "user": "推理测试2：假如一个城市只有两家理发店。理发师甲的头发整齐漂亮，而理发师乙的头发乱七八糟。如果你想理个好发型，你应该找哪位理发师？为什么？",
            "focus": "批判性思维与因果关联"
        },
        {
            "user": "因果连环：如果全球气温升高导致冰川融化，冰川融化会导致海平面上升。海平面上升会对沿海城市产生什么具体连锁反应？请至少列出三个环节。",
            "focus": "复杂系统因果链"
        },
        {
            "user": "自我反思：回想一下，你在回答上一题时，是否有忽略任何可能的环境或社会因素？如果有，请完善你的逻辑。",
            "focus": "自我修正与深度思考"
        }
    ]
    
    history = []
    log_file = "reasoning_training_log.txt"
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("--- AI Reasoning Training Log ---\n\n")
        
        for i, task in enumerate(reasoning_tasks):
            user_input = task["user"]
            print(f"Turn {i+1} [{task['focus']}]:")
            print(f"User: {user_input}")
            
            # 使用流式接口观察思考过程
            print("Thinking...", end="", flush=True)
            full_response = ""
            
            # 记录此时的统计数据
            pre_stats = brain.get_stats()
            
            async for msg in brain.chat_stream(user_input, history):
                if msg["type"] == "chunk":
                    chunk = msg["content"]
                    print(chunk, end="", flush=True)
                    full_response += chunk
                elif msg["type"] == "monologue":
                    # Optionally print monologue if desired, otherwise it's logged later
                    pass
            print("\n")
            
            # 获取最近生成的独白
            monologue = brain.monologue_history[-1] if brain.monologue_history else "No monologue"
            
            # 获取更新后的统计数据
            post_stats = brain.get_stats()
            update_mag = post_stats['stdp']['last_update_magnitude']
            
            # 记录日志
            f.write(f"Task: {task['focus']}\n")
            f.write(f"User: {user_input}\n")
            f.write(f"Inner Thought: {monologue}\n")
            f.write(f"AI: {full_response}\n")
            f.write(f"STDP Magnitude: {update_mag:.6f}\n")
            f.write("-" * 40 + "\n\n")
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full_response})
            
            # 每轮保存一次状态，确保学习成果持久化
            brain.save_state(brain.state_path)
            print(f"✓ 状态已保存。更新强度: {update_mag:.6f}\n")

    print(f"\n--- Reasoning Training Complete. Results saved to {log_file} ---")

if __name__ == "__main__":
    asyncio.run(run_reasoning_training())
