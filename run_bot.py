#!/usr/bin/env python3
"""
启动 Telegram Bot 服务
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/z/my-project/stdpbrain')
os.chdir('/home/z/my-project/stdpbrain')

from configs.arch_config import BrainAIConfig
from core.interfaces import BrainAIInterface
from telegram_bot.bot import BrainAIBot

def main():
    print("=" * 60)
    print("类人脑双系统全闭环 AI 架构 - Telegram Bot")
    print("=" * 60)
    
    # 初始化配置
    config = BrainAIConfig()
    config.model_path = "./models/Qwen3.5-0.8B"
    
    print("\n[初始化] 加载模型和模块...")
    
    try:
        # 创建 AI 实例
        ai = BrainAIInterface(config, device="cpu")
        print("[初始化] AI 接口加载完成 ✓")
    except Exception as e:
        print(f"[错误] AI 初始化失败：{e}")
        print("\n[提示] 将以测试模式启动 Bot（无 AI 模型）")
        ai = None
    
    # Telegram Bot Token
    bot_token = "8533918353:AAG6Pxr0A4C4kJpCVjYzbtwtFzN4KZCcRag"
    
    # 创建 Bot
    bot = BrainAIBot(
        token=bot_token,
        ai_interface=ai,
        stream_chunk_size=1,
        stream_delay_ms=50
    )
    
    print(f"\n[Bot] Token: {bot_token[:20]}...")
    print("[Bot] 启动中...")
    print("\n按 Ctrl+C 停止 Bot")
    print("=" * 60)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n[Bot] 已停止")
    except Exception as e:
        print(f"[错误] Bot 运行失败：{e}")

if __name__ == "__main__":
    main()
