"""Telegram Bot 流式输出处理器

核心功能:
- 实时流式生成响应
- Token-by-token 输出
- 打字状态模拟
- 真实 Qwen 模型集成
"""

import asyncio
import time
from typing import Callable, Optional, List, Dict, Any, AsyncGenerator


class StreamHandler:
    """
    流式输出处理器
    
    支持:
    - 逐 token 流式生成
    - 打字状态显示
    - 真实模型推理
    - 错误处理和回调
    """
    
    def __init__(
        self,
        ai_interface=None,
        delay_ms: int = 50,
        chunk_size: int = 1,
        max_tokens: int = 500
    ):
        """
        初始化流式处理器
        
        Args:
            ai_interface: BrainAIInterface 实例
            delay_ms: token 间延迟 (毫秒)
            chunk_size: 每次输出的 chunk 大小
            max_tokens: 最大生成 token 数
        """
        self.ai = ai_interface
        self.delay_ms = delay_ms
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
      
        # 回调函数
        self.on_token_callback: Optional[Callable] = None
        self.on_complete_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
      
        # 统计信息
        self.total_streams = 0
        self.total_tokens_generated = 0
        self.avg_stream_length = 0.0
    
    async def generate_stream(
        self,
        input_text: str,
        context: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成响应
        
        Args:
            input_text: 输入文本
            context: 上下文信息
        
        Yields:
            chunk: 文本块
        """
        start_time = time.time()
        full_response = ""
        
        # ========== 1. 使用真实 AI 接口流式生成 ==========
        if self.ai is not None:
            # 使用 BrainAIInterface 的流式生成
            async for token_text in self.ai.generate_stream(
                input_text=input_text,
                max_tokens=self.max_tokens,
                **kwargs
            ):
                full_response += token_text
                yield token_text
                
                # 模拟打字延迟
                if self.delay_ms > 0:
                    await asyncio.sleep(self.delay_ms / 1000.0)
            
            self.total_tokens_generated += len(full_response.split())
        else:
            # ========== 2. 降级：使用简化实现 ==========
            full_response = self._generate_simple_response(input_text)
            
            # 按字符分块输出
            chunk_size = 3
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i+chunk_size]
                yield chunk
                await asyncio.sleep(self.delay_ms / 1000.0)
        
        # ========== 3. 完成回调 ==========
        if self.on_complete_callback:
            self.on_complete_callback(full_response)
        
        elapsed = time.time() - start_time
        self.total_streams += 1
        self.avg_stream_length = (
            (self.avg_stream_length * (self.total_streams - 1) + len(full_response))
            / self.total_streams
        )

    def _generate_simple_response(self, input_text: str) -> str:
        """生成简单响应 (无模型时使用)"""
        responses = {
            "你好": "你好！我是类人脑 AI 助手，基于 Qwen3.5-2B 模型。我有什么可以帮助你的吗？",
            "介绍": "我是基于海马体 - 新皮层双系统架构的类人脑 AI，具有 100Hz 高刷新推理、STDP 在线学习等特性。",
            "默认": "收到你的消息了！我正在使用类人脑架构进行处理，支持流式输出和实时交互。"
        }
        
        for key, value in responses.items():
            if key in input_text:
                return value
        
        return f"测试回复：{input_text}。实际使用时会连接真实的 Qwen3.5-2B 模型。"

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'total_streams': self.total_streams,
            'total_tokens_generated': self.total_tokens_generated,
            'avg_stream_length': self.avg_stream_length
        }


class TypingSimulator:
    """打字模拟器"""
    
    def __init__(self, bot, chat_id):
        self.bot = bot
        self.chat_id = chat_id
        self._is_typing = False
        self._task = None

    async def _typing_loop(self):
        while self._is_typing:
            await self.bot.send_chat_action(chat_id=self.chat_id, action="typing")
            await asyncio.sleep(4) # Telegram typing state lasts 5 seconds

    async def start_typing(self):
        if not self._is_typing:
            self._is_typing = True
            self._task = asyncio.create_task(self._typing_loop())

    async def stop_typing(self):
        self._is_typing = False
        if self._task:
            self._task.cancel()
            self._task = None


if __name__ == "__main__":
    # 测试流式处理器
    async def test_stream():
        handler = StreamHandler()
        
        print("=" * 60)
        print("流式处理器测试")
        print("=" * 60)
        
        test_inputs = ["你好", "介绍一下你自己"]
        
        for input_text in test_inputs:
            print(f"\n输入：{input_text}")
            print("输出：", end="", flush=True)
            
            full_response = ""
            async for chunk in handler.generate_stream(input_text):
                print(chunk, end="", flush=True)
                full_response += chunk
        
        print(f"\n\n完整响应：{full_response}")
        print(f"统计：{handler.get_stats()}")
    
    asyncio.run(test_stream())
