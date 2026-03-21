"""验证目标系统修复"""

from core.interfaces import BrainAIInterface
from configs.arch_config import BrainAIConfig
import time

print("\n=== 目标系统验证测试 ===\n")

config = BrainAIConfig()
ai = BrainAIInterface(config)

# 测试目标推断
test_inputs = [
    "我叫张三，我来自北京",
    "你还记得我叫什么吗？"
]

for user_input in test_inputs:
    print(f"\n用户: {user_input}")
    
    # 推断目标
    if ai.goal_system:
        goal = ai.goal_system.infer_goal(user_input, ai.current_thought_state)
        print(f"推断目标: {goal.goal_type.value}")
        print(f"目标描述: {goal.description}")
    
    # 获取回复
    start = time.time()
    response = ai.chat(user_input, max_tokens=60)
    elapsed = (time.time() - start) * 1000
    
    print(f"AI: {response[:80]}...")
    print(f"[{elapsed:.0f}ms]")
    
    # 显示目标状态
    if ai.goal_system:
        goal_info = ai.goal_system.get_current_goal_info()
        reward = ai.goal_system.get_reward_signal()
        print(f"[内在奖励: {reward:.3f}]")
        if 'type' in goal_info:
            print(f"[当前目标: {goal_info['type']} - {goal_info['description'][:40]}]")

print("\n=== 验证完成 ===\n")
