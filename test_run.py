#!/usr/bin/env python3
"""
类人脑AI架构 - 简化测试脚本

不依赖 torch，用于验证核心逻辑和流程
"""

import sys
import time
import random


class SimpleBrainAI:
    """简化版类人脑AI (无需 torch)"""
    
    def __init__(self):
        print("=" * 60)
        print("类人脑双系统全闭环 AI架构 - 简化测试版")
        print("=" * 60)
        
        # 响应模板
        self.responses = {
            "你好": [
                "你好！我是类人脑AI 助手，基于海马体 - 新皮层双系统架构。",
                "你好！我支持 100Hz 高刷新推理和 STDP 在线学习。"
            ],
            "介绍": [
                "我是基于Qwen3.5-0.8B 的类人脑AI，具有海马体记忆系统。",
                "我采用双权重架构：90% 静态 +10% 动态，支持终身学习。"
            ],
            "STDP": [
                "STDP(脉冲时序依赖可塑性) 是我的核心学习机制。",
                "通过 STDP，我能够'推理即学习'，无需反向传播。"
            ],
            "架构": [
                "我的架构包括：海马体 (EC-DG-CA3-CA1-SWR)、STDP 引擎、100Hz 刷新引擎。",
                "我采用类脑设计，O(1) 复杂度注意力，显存占用≤420MB。"
            ],
            "Telegram": [
                "我支持 Telegram Bot，可以流式输出和实时交互。",
                "使用'main.py --mode telegram'即可启动 Bot。"
            ]
        }
        
        self.cycle_count = 0
        print("\n✓ 初始化完成，准备就绪\n")
    
    def generate(self, text: str, max_tokens: int = 100) -> str:
        """生成响应"""
        start_time = time.time()
        
        # 匹配关键词
        text_lower = text.lower()
        response = None
        
        for keyword, replies in self.responses.items():
            if keyword in text_lower and keyword != "默认":
                response = random.choice(replies)
                break
        
        if not response:
            response = f"收到：{text[:50]}。这是一个测试响应，实际使用时会连接真实的语言模型。"
        
        # 模拟推理延迟
        time.sleep(0.1)
        self.cycle_count += len(text)
        
        elapsed = (time.time() - start_time) * 1000
        
        return f"{response}\n\n[推理耗时：{elapsed:.1f}ms, 周期数：{self.cycle_count}]"
    
    def chat(self, message: str, history=None) -> str:
        """对话接口"""
        return self.generate(message)
    
    def get_stats(self) -> dict:
        """获取统计"""
        return {
            'cycle_count': self.cycle_count,
            'system': 'simplified',
            'status': 'running'
        }


def test_basic_chat():
    """测试基本对话"""
    print("\n" + "=" * 60)
    print("测试 1: 基本对话")
    print("=" * 60)
    
    ai = SimpleBrainAI()
    
    test_cases = [
        "你好",
        "介绍一下你自己",
        "什么是 STDP",
        "你的架构是什么",
        "支持 Telegram 吗"
    ]
    
    for test in test_cases:
        print(f"\n你：{test}")
        response = ai.chat(test)
        print(f"AI: {response}")
    
    print("\n✓ 对话测试完成")
    return True


def test_context():
    """测试上下文"""
    print("\n" + "=" * 60)
    print("测试 2: 多轮对话上下文")
    print("=" * 60)
    
    ai = SimpleBrainAI()
    history = []
    
    conversation = [
        "你好",
        "你能做什么",
        "介绍一下海马体",
        "谢谢"
    ]
    
    for msg in conversation:
        print(f"\n你：{msg}")
        response = ai.chat(msg, history)
        print(f"AI: {response}")
        
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": response})
    
    print("\n✓ 上下文测试完成")
    return True


def test_stats():
    """测试统计功能"""
    print("\n" + "=" * 60)
    print("测试 3: 系统统计")
    print("=" * 60)
    
    ai = SimpleBrainAI()
    
    # 生成一些内容
    for i in range(5):
        ai.chat(f"测试消息{i}")
    
    stats = ai.get_stats()
    
    print("\n系统统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ 统计测试完成")
    return True


def main():
    """主测试函数"""
    print("\n类人脑AI架构 - 功能测试套件\n")
    
    all_passed = True
    
    try:
        all_passed &= test_basic_chat()
        all_passed &= test_context()
        all_passed &= test_stats()
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ 所有测试通过!")
        else:
            print("❌ 部分测试失败")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n提示:")
    print("  - 这是简化测试版本，不依赖 torch")
    print("  - 完整版本需要安装：pip install torch python-telegram-bot")
    print("  - 完整功能请参考 README.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
