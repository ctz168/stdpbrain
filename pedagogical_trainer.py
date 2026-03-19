import asyncio
import json
import os
import torch
import logging
import sys
from core.interfaces import BrainAIInterface
from configs.arch_config import BrainAIConfig

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PedagogicalTrainer")

LOCAL_DATASET = "chinese_reasoning_data.jsonl"
STATE_PATH = "brain_state.pt"

class PedagogicalTrainer:
    def __init__(self, brain: BrainAIInterface):
        self.brain = brain
        self.history = []
        self.training_log = "pedagogical_training_log.txt"

    async def speak(self, text: str, role: str = "user"):
        """模拟人与 AI 的对话，并实时打印 AI 的思考过程"""
        print(f"\n[{role.upper()}]: {text}")
        if role == "user":
            self.history.append({"role": "user", "content": text})
            
            full_response = ""
            print("[THINKING...]", flush=True)
            
            async for msg in self.brain.chat_stream(text, self.history[:-1]):
                if msg["type"] == "monologue":
                    # 打印内心独白
                    print(f"\n >>> THOUGHT: {msg['content']}")
                    print(" >>> RESPONSE: ", end="", flush=True)
                elif msg["type"] == "chunk":
                    print(msg["content"], end="", flush=True)
                    full_response += msg["content"]
            
            print("\n")
            self.history.append({"role": "assistant", "content": full_response})
            return full_response

    async def run_priming(self):
        """第一阶段：场面话，进入状态"""
        print("\n--- Phase 1: Priming ---\n")
        await self.speak("孩子，我们要开始今天的逻辑与推理训练了。这不仅是回答问题，更是训练你如何在大脑中构建清晰的逻辑链条。你准备好进入深度学习状态了吗？")
        await self.speak("很好。在接下来的练习中，如果你发现自己的逻辑有误，不要气馁，通过反思和调整，你的突触连接会变得更加稳固。我们开始吧。")

    async def run_reasoning_tasks(self):
        """第二阶段：启发式教学"""
        print("\n--- Phase 2: Pedagogical Training ---\n")
        
        if not os.path.exists(LOCAL_DATASET):
            print(f"[ERROR] 未找到数据集文件: {LOCAL_DATASET}")
            return

        tasks = []
        with open(LOCAL_DATASET, "r", encoding="utf-8") as f:
            for line in f: tasks.append(json.loads(line))

        for i, item in enumerate(tasks):
            q, a, r = item["question"], item["answer"], item["reasoning"]
            print(f"--- [Task {i+1}/15] ---")
            
            # 1. 提问
            ai_ans = await self.speak(q)
            
            # 2. 判断（是非准绳）
            if a in ai_ans:
                await self.speak(f"做得好！你的逻辑非常准确，答案正是 {a}。继续保持这种清晰的思考。")
                self.brain.model.set_reward(2.0)
            else:
                # 3. 启发式纠错 (Socratic hint)
                print(f"  [WARN] 回答不够准确。寻找逻辑漏洞...")
                self.brain.model.set_reward(-1.0)
                
                # 给出启发式提示，而不是直接给答案
                hint = f"再仔细想一想，题目中提到的关键信息是'{a}'相关的逻辑。你觉得你在推理过程中是否有忽略掉什么细节？（提示：请关注 {r} 的逻辑）"
                ai_retry = await self.speak(hint)
                
                # 4. 第二次尝试后给予总结
                if a in ai_retry:
                    await self.speak(f"非常棒！你通过自我反思修正了逻辑。这就是 STDP 机制在你大脑中发挥作用的过程。")
                    self.brain.model.set_reward(1.5)
                else:
                    await self.speak(f"没关系。正确的推理逻辑是：{r}。最终答案应该是 {a}。请对比一下你刚才的思考过程，看看差异在哪里，并将其固化到你的长时记忆中。")
                    self.brain.model.set_reward(1.0) # 虽然没答对，但学习了正确过程，给予基础奖励
            
            # 5. 持久化
            if (i+1) % 3 == 0:
                self.brain.save_state(STATE_PATH)
                print(f"[OK] 进度已固化至: {STATE_PATH}")

async def main():
    config = BrainAIConfig()
    config.model_path = "./models/Qwen3.5-0.8B"
    config.stdp.enabled = True
    brain = BrainAIInterface(config)
    brain.load_state(STATE_PATH)
    
    trainer = PedagogicalTrainer(brain)
    
    # 执行流水线
    await trainer.run_priming()
    await trainer.run_reasoning_tasks()
    
    brain.save_state(STATE_PATH)
    print("\n=== 深度启发式训练任务全部完成 ===")

if __name__ == "__main__":
    asyncio.run(main())
