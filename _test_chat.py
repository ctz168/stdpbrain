"""
测试优化后的AI对话系统
测试目标：
1. AI能正常回应
2. 思维链更理性，减少情绪化
3. 推理能力提升
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.arch_config import BrainAIConfig
from core.interfaces import BrainAIInterface

def run_chat_test():
    """运行3轮对话测试"""
    print("=" * 60)
    print("       AI对话系统优化测试")
    print("=" * 60)
    print()
    
    # 初始化配置
    config = BrainAIConfig()
    
    # 初始化AI
    print("[初始化] 正在加载AI系统...")
    ai = BrainAIInterface(config)
    print("[初始化] 完成!")
    print()
    
    # 测试问题列表 - 涵盖不同类型
    test_questions = [
        "如果明天下雨，我应该带什么？请用逻辑推理回答。",
        "分析一下：为什么学习编程很重要？请分点说明。",
        "什么是质数？请给我举几个例子并解释原因。"
    ]
    
    # 存储对话历史
    history = []
    
    # 进行3轮对话
    for i, question in enumerate(test_questions, 1):
        print("=" * 60)
        print(f"[第 {i} 轮对话]")
        print("-" * 60)
        print(f"用户: {question}")
        print("-" * 60)
        
        try:
            # 获取AI响应
            response = ai.chat(question, history=history, max_tokens=200)
            
            print(f"AI: {response}")
            print("-" * 60)
            
            # 添加到历史
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": response})
            
            # 评估响应质量
            print("[评估]")
            has_structure = any(kw in response for kw in ["首先", "其次", "第一", "第二", "因为", "所以", "因此", "分析", "结论"])
            has_logic = any(kw in response for kw in ["因为", "所以", "因此", "导致", "原因", "结果", "推导"])
            not_emotional = not any(kw in response for kw in ["感觉", "情绪", "心情", "忽然", "飘向", "沉思"])
            
            print(f"  - 有结构化表达: {'✓' if has_structure else '✗'}")
            print(f"  - 有逻辑推理: {'✓' if has_logic else '✗'}")
            print(f"  - 非情绪化: {'✓' if not_emotional else '✗'}")
            
        except Exception as e:
            print(f"[错误] 对话失败: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # 保存状态
    print("=" * 60)
    print("[结束] 正在保存AI状态...")
    ai.save_state("brain_state.pt")
    print("[结束] 测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    run_chat_test()
