#!/usr/bin/env python3
"""
Telegram Bot 启动脚本

用法:
    python telegram_bot/run.py

环境变量:
    TELEGRAM_BOT_TOKEN: Bot Token (可选，也可在配置文件中设置)
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="类人脑AI Telegram Bot")
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Telegram Bot Token (默认使用配置文件中的值)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/Qwen3.5-0.8B-Base",
        help="Qwen3.5-0.8B 模型路径"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="运行设备"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="流式输出块大小 (默认：1)"
    )
    
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=50,
        help="流式延迟 (毫秒，默认：50)"
    )
    
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="启用异步模式"
    )
    
    return parser.parse_args()


def load_config():
    """加载配置文件"""
    from configs.arch_config import BrainAIConfig
    return BrainAIConfig()


async def run_async_bot(bot):
    """运行异步 Bot"""
    try:
        await bot.start_async()
        
        # 保持运行
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n正在停止 Bot...")
    finally:
        if bot.application:
            await bot.application.stop()
            await bot.application.shutdown()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("类人脑AI - Telegram Bot")
    print("=" * 60)
    
    # ========== 1. 获取 Bot Token ==========
    bot_token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token:
        # 使用硬编码的 token (仅用于测试)
        print("[警告] 未提供 Bot Token，使用默认 Token")
        bot_token = "7983263905:AAFsMuGRdZzWv7KfUaAkJocu0l7LsHrScuc"
    
    print(f"\n[Bot] Token: {bot_token[:20]}...")
    
    # ========== 2. 初始化配置 ==========
    print("\n[初始化] 加载配置...")
    config = load_config()
    config.model_path = args.model_path
    
    # ========== 3. 创建 AI 实例 ==========
    print("[初始化] 加载 AI 模型...")
    
    try:
        from core.interfaces import BrainAIInterface
        
        ai = BrainAIInterface(config, device=args.device)
        print("[初始化] AI 模型加载完成 ✓")
        
    except Exception as e:
        print(f"[警告] AI 模型加载失败：{e}")
        print("  Bot 将以测试模式运行 (无真实 AI 响应)")
        ai = None
    
    # ========== 4. 创建 Bot 实例 ==========
    print("\n[初始化] 创建 Telegram Bot...")
    
    from telegram_bot.bot import BrainAIBot
    
    bot = BrainAIBot(
        token=bot_token,
        ai_interface=ai,
        stream_chunk_size=args.chunk_size,
        stream_delay_ms=args.delay_ms
    )
    
    print("[初始化] Bot 创建完成 ✓")
    
    # ========== 5. 显示帮助信息 ==========
    print("\n" + "=" * 60)
    print("Bot 已就绪!")
    print("=" * 60)
    print(f"\nBot Token: {bot_token[:15]}...")
    print(f"流式块大小：{args.chunk_size}")
    print(f"流式延迟：{args.delay_ms}ms")
    print(f"运行模式：{'异步' if args.async_mode else '同步'}")
    print("\n按 Ctrl+C 停止 Bot")
    print("=" * 60)
    
    # ========== 6. 运行 Bot ==========
    try:
        if args.async_mode:
            # 异步模式
            asyncio.run(run_async_bot(bot))
        else:
            # 同步模式
            bot.run()
    
    except KeyboardInterrupt:
        print("\n\n收到停止信号，正在退出...")
    except Exception as e:
        print(f"\n[错误] Bot 运行失败：{e}")
        sys.exit(1)
    
    print("\nBot 已停止")


if __name__ == "__main__":
    main()
