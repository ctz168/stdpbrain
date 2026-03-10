#!/usr/bin/env python3
import argparse, sys, os

def parse_args():
 parser= argparse.ArgumentParser(description="Telegram Bot")
 parser.add_argument("--token", type=str, required=True)
 parser.add_argument("--model-path", type=str, default="./models/Qwen3.5-0.8B-Base")
 parser.add_argument("--device", type=str, default="cpu")
 parser.add_argument("--quantization", type=str, default="INT4")
 return parser.parse_args()

def main():
   args= parse_args()
 print("=" * 60)
 print("Telegram Bot - Qwen 模型版")
 print("=" * 60)
 if not os.path.exists(args.model_path):
     print(f"\n❌ 模型未找到: {args.model_path}\n请运行：huggingface-cli download Qwen/Qwen3.5-0.8B-Base --local-dir ./models/Qwen3.5-0.8B-Base")
     sys.exit(1)
 print(f"\n加载 Qwen 模型...")
 try:
     from core.qwen_interface import create_real_qwen_ai
       ai = create_real_qwen_ai(model_path=args.model_path, device=args.device, quantization=args.quantization)
     print("✓ Qwen 模型加载成功")
 except Exception as e:
     print(f"❌ 失败：{e}")
     sys.exit(1)
 print(f"\n初始化 Bot...")
 try:
     from telegram_bot.bot import BrainAIBot
       bot = BrainAIBot(token=args.token, ai_interface=ai)
     print("✓ Bot 就绪")
 except Exception as e:
     print(f"❌ 失败：{e}")
     sys.exit(1)
 print(f"\n启动服务... (Ctrl+C 停止)")
 print("=" * 60)
  bot.run()

if __name__ == "__main__":
   main()
