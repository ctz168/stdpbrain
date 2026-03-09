#!/usr/bin/env python3
"""
Telegram Bot 测试脚本

测试流式输出和基本功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_stream_handler():
    """测试流式处理器"""
    print("=" * 60)
    print("测试流式处理器")
    print("=" * 60)
    
    from telegram_bot.stream_handler import StreamHandler
    
    # 创建模拟 AI 接口
    class MockAI:
        def __init__(self):
            self.self_loop = None
    
    mock_ai = MockAI()
    handler = StreamHandler(
        ai_interface=mock_ai,
        chunk_size=2,
        delay_ms=100,
        max_tokens=50
    )
    
    # 测试流式生成
    test_input = "你好，请介绍一下自己"
    print(f"\n输入：{test_input}\n")
    print("流式输出:")
    print("-" * 40)
    
    full_response = ""
    async for chunk in handler.generate_stream(test_input):
        print(chunk, end='', flush=True)
        full_response += chunk
    
    print("\n" + "-" * 40)
    print(f"\n完整响应长度：{len(full_response)}")
    print("✓ 流式处理器测试完成\n")


async def test_typing_simulator():
    """测试打字模拟器"""
    print("=" * 60)
    print("测试打字模拟器")
    print("=" * 60)
    
    from telegram_bot.stream_handler import TypingSimulator
    
    # 创建模拟 Bot
    class MockBot:
        async def send_chat_action(self, chat_id, action):
            print(f"[MockBot] 发送 {action} 状态到聊天 {chat_id}")
    
    mock_bot = MockBot()
    typing = TypingSimulator(mock_bot, chat_id=123456)
    
    print("\n开始打字状态...")
    await typing.start_typing()
    
    print("模拟打字中...")
    await asyncio.sleep(2)
    
    print("停止打字状态...")
    await typing.stop_typing()
    
    print("✓ 打字模拟器测试完成\n")


def test_bot_initialization():
    """测试 Bot 初始化"""
    print("=" * 60)
    print("测试 Bot 初始化")
    print("=" * 60)
    
    from telegram_bot.bot import BrainAIBot
    
    token = "8533918353:AAG6Pxr0A4C4kJpCVjYzbtwtFzN4KZCcRag"
    
    print(f"\n使用 Token: {token[:20]}...")
    
    bot = BrainAIBot(
        token=token,
        ai_interface=None,  # 无 AI 接口 (测试模式)
        stream_chunk_size=1,
        stream_delay_ms=50
    )
    
    print(f"Bot 已创建")
    print(f"  - Chunk Size: {bot.stream_chunk_size}")
    print(f"  - Delay MS: {bot.stream_delay_ms}")
    print(f"  - 最大上下文：{bot.max_context_length}")
    print("✓ Bot 初始化测试完成\n")


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Telegram Bot 测试套件")
    print("=" * 60 + "\n")
    
    try:
        # 测试 1: Bot 初始化
        test_bot_initialization()
        
        # 测试 2: 流式处理器
        await test_stream_handler()
        
        # 测试 3: 打字模拟器
        await test_typing_simulator()
        
        print("=" * 60)
        print("所有测试通过 ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
