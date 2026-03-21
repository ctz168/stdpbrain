"""
聊天功能测试脚本
自动化测试多个对话场景
"""

from core.interfaces import BrainAIInterface
from configs.arch_config import BrainAIConfig
import time

def test_chat():
    print("\n" + "=" * 60)
    print("  聊天功能实际测试")
    print("=" * 60)
    
    # 初始化系统
    print("\n[1] 初始化系统...")
    config = BrainAIConfig()
    ai = BrainAIInterface(config)
    
    # 测试场景列表
    test_scenarios = [
        ("自我介绍", "你好，请介绍一下你自己。"),
        ("身份认知", "你是谁创造的？"),
        ("记忆注入", "我叫张三，我来自北京，是一名软件工程师。"),
        ("无关对话", "今天天气怎么样？"),
        ("记忆调用", "你还记得我叫什么名字吗？"),
        ("能力展示", "请用逻辑推理的方式分析：如果明天下雨，我应该做什么准备？"),
        ("复杂推理", "1+1=2, 2+2=4, 4+4=8, 请问8+8=?"),
    ]
    
    history = []
    
    for i, (scenario_name, user_input) in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"测试场景 {i}: {scenario_name}")
        print(f"{'='*60}")
        print(f"用户: {user_input}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取AI回复
        response = ai.chat(user_input, history, max_tokens=200)
        
        # 计算耗时
        elapsed = (time.time() - start_time) * 1000
        
        print(f"\nAI: {response}")
        print(f"\n[耗时: {elapsed:.1f}ms]")
        
        # 更新历史
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        
        # 获取当前目标状态（如果有）
        if hasattr(ai, 'goal_system') and ai.goal_system:
            goal_info = ai.goal_system.get_current_goal_info()
            reward = ai.goal_system.get_reward_signal()
            print(f"[内在奖励: {reward:.3f}]")
            if goal_info and 'type' in goal_info:
                print(f"[当前目标: {goal_info.get('type', 'N/A')} - {goal_info.get('description', 'N/A')[:50]}]")
    
    # 显示系统统计
    print(f"\n{'='*60}")
    print("  系统统计信息")
    print(f"{'='*60}")
    
    stats = ai.get_stats()
    print(f"\n[海马体系统]")
    print(f"  记忆数量: {stats['hippocampus'].get('num_memories', 'N/A')}")
    print(f"  内存使用: {stats['hippocampus'].get('memory_usage_mb', 0):.2f}MB")
    
    print(f"\n[STDP 引擎]")
    print(f"  总更新次数: {stats['stdp'].get('total_updates', 0)}")
    print(f"  动态权重范数: {stats['stdp'].get('dynamic_weight_norm', 0):.6f}")
    
    print(f"\n[独白系统]")
    print(f"  独白历史数: {stats['monologue'].get('history_count', 0)}")
    
    print(f"\n[系统]")
    print(f"  总周期数: {stats['system'].get('total_cycles', 0)}")
    print(f"  设备: {stats['system'].get('device', 'N/A')}")
    
    # 保存状态
    print(f"\n{'='*60}")
    print("  保存状态")
    print(f"{'='*60}")
    ai.save_state("brain_state.pt")
    print("状态已保存到 brain_state.pt")
    
    print(f"\n{'='*60}")
    print("  测试完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_chat()
