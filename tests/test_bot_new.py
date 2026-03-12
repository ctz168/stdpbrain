
import asyncio
import sys
import os

# 将项目根目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.arch_config import BrainAIConfig
from core.interfaces import BrainAIInterface
from telegram_bot.bot import BrainAIBot

async def test_bot_logic():
    print("=" * 60)
    print("测试类人脑 AI Bot 新功能")
    print("=" * 60)
    sys.stdout.flush()
    
    # 1. 初始化配置 (使用简化模型)
    config = BrainAIConfig()
    config.model_path = "" # 强制使用简化模型
    
    # 2. 创建 AI 实例
    ai = BrainAIInterface(config)
    
    # 3. 创建 Bot 实例
    bot = BrainAIBot(token="TEST_TOKEN", ai_interface=ai)
    
    # 4. 测试后台思考循环
    print("\n[测试] 启动后台思考流...")
    sys.stdout.flush()
    # 手动启动一次思考循环
    stats = ai.think()
    print(f"✓ 思考完成，系统统计：{stats['system']}")
    sys.stdout.flush()
    
    # 5. 测试思考过程生成
    print("\n[测试] 思考过程生成...")
    sys.stdout.flush()
    thought = ai._generate_thought_process("你好")
    print(f"生成的思考过程：\n{thought}")
    sys.stdout.flush()
    
    # 6. 测试流式生成 (模拟)
    print("\n[测试] 流式生成模拟...")
    sys.stdout.flush()
    async for chunk in ai.generate_stream("介绍一下你自己"):
        print(chunk, end="", flush=True)
    print("\n✓ 流式生成完成")
    sys.stdout.flush()
    
    print("\n" + "=" * 60)
    print("所有功能测试通过！")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_bot_logic())
