"""改进的目标驱动测试"""

from core.interfaces import BrainAIInterface
from configs.arch_config import BrainAIConfig
import time

print("\n=== 目标驱动系统改进测试 ===\n")

config = BrainAIConfig()
ai = BrainAIInterface(config)

# 测试不同目标类型
test_cases = [
    {
        "input": "我叫张三，今年25岁",
        "expected_type": "remember",
        "expected_response": "好的，我记住了"
    },
    {
        "input": "你还记得我叫什么吗？",
        "expected_type": "recall", 
        "expected_response": "您叫张三"
    }
]

for i, test_case in enumerate(test_cases, 1):
    user_input = test_case["input"]
    expected_type = test_case["expected_type"]
    
    print(f"\n{'='*60}")
    print(f"测试 {i}: {user_input}")
    print(f"{'='*60}")
    
    # 推断目标
    if ai.goal_system:
        goal = ai.goal_system.infer_goal(user_input, ai.current_thought_state)
        print(f"[OK] 目标类型: {goal.goal_type.value}")
        print(f"[OK] 目标描述: {goal.description}")
        
        # 验证目标类型
        if goal.goal_type.value == expected_type:
            print(f"[OK] 目标推断正确")
        else:
            print(f"[ERROR] 目标推断错误：期望 {expected_type}，实际 {goal.goal_type.value}")
    
    # 获取回复
    print(f"\n生成回复中...")
    start = time.time()
    response = ai.chat(user_input, max_tokens=80)
    elapsed = (time.time() - start) * 1000
    
    print(f"\nAI: {response[:120]}")
    print(f"\n[耗时: {elapsed:.0f}ms]")
    
    # 显示目标状态和奖励
    if ai.goal_system:
        # 先更新目标进度
        if ai.goal_system.current_goal:
            ai.goal_system.update_progress(0.8)  # 暂时设为80%完成
        
        goal_info = ai.goal_system.get_current_goal_info()
        reward = ai.goal_system.get_reward_signal()
        print(f"[内在奖励: {reward:.3f}]")
        if 'progress' in goal_info:
            print(f"[目标进度: {goal_info['progress']*100:.1f}%]")

print(f"\n{'='*60}")
print("测试完成")
print(f"{'='*60}\n")

# 保存状态
ai.save_state("brain_state.pt")
print("状态已保存\n")
