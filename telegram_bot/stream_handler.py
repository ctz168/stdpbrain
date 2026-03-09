"""
流式输出处理器

实现类人脑AI 模型的流式生成，支持 Telegram Bot 的实时输出
"""

import asyncio
from typing import AsyncGenerator, Optional, Callable, List
import time


class StreamHandler:
    """
    流式输出处理器
    
    功能:
    - 将模型生成分解为 token 级别的流式输出
    - 支持实时推送生成的内容到 Telegram
    - 支持打字状态显示
    """
    
    def __init__(
        self,
        ai_interface,
        chunk_size: int = 1,      # 每次输出的 token 数
        delay_ms: int = 50,       # 输出间隔 (毫秒)
        max_tokens: int = 200     # 最大生成 token 数
    ):
        """
        初始化流式处理器
        
        Args:
            ai_interface: BrainAIInterface 实例
            chunk_size: 每次输出的 token 数量
            delay_ms: 输出延迟 (毫秒)
            max_tokens: 最大生成 token 数
        """
        self.ai = ai_interface
        self.chunk_size = chunk_size
        self.delay_ms = delay_ms
        self.max_tokens = max_tokens
        
        # 回调函数
        self.on_token_callback: Optional[Callable[[str], None]] = None
        self.on_complete_callback: Optional[Callable[[str], None]] = None
        self.on_error_callback: Optional[Callable[[Exception], None]] = None
    
    async def generate_stream(
        self,
        input_text: str,
        context: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        生成流式输出
        
        Args:
            input_text: 输入文本
            context: 上下文列表
        
        Yields:
            token: 生成的文本片段
        """
        full_response = ""
        start_time = time.time()
        
        try:
            # 使用自闭环优化器生成
            if hasattr(self.ai, 'self_loop') and self.ai.self_loop:
                result = self.ai.self_loop.run(input_text)
                optimized_input = result.output_text
            else:
                optimized_input = input_text
            
            # Tokenize 输入
            input_tokens = self._tokenize(optimized_input)
            
            # 模拟流式生成 (实际应调用模型)
            # TODO: 替换为真实的模型流式推理
            generated_tokens = []
            
            for i, token_id in enumerate(input_tokens[:self.max_tokens]):
                # 模拟推理延迟
                await asyncio.sleep(self.delay_ms / 1000.0)
                
                # 生成 token (简化处理)
                token = self._detokenize_single(token_id)
                generated_tokens.append(token)
                
                # 累积输出
                full_response += token
                
                # 每 chunk_size 个 token 输出一次
                if len(generated_tokens) % self.chunk_size == 0:
                    chunk = "".join(generated_tokens[-self.chunk_size:])
                    yield chunk
                    
                    # 触发回调
                    if self.on_token_callback:
                        self.on_token_callback(chunk)
            
            # 如果没有生成内容，使用简化响应
            if not full_response:
                full_response = self._generate_simple_response(input_text)
                # 分块输出
                chunk_size = 20
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i:i+chunk_size]
                    yield chunk
                    await asyncio.sleep(self.delay_ms / 1000.0)
                    
                    if self.on_token_callback:
                        self.on_token_callback(chunk)
            
            # 完成回调
            if self.on_complete_callback:
                self.on_complete_callback(full_response)
            
            elapsed = time.time() - start_time
            print(f"[Stream] 生成完成，耗时：{elapsed*1000:.1f}ms, 长度：{len(full_response)}")
            
        except Exception as e:
            print(f"[Stream] 错误：{e}")
            if self.on_error_callback:
                self.on_error_callback(e)
            yield f"[错误] 生成失败：{str(e)}"
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize 输入"""
        # TODO: 使用真实 tokenizer
        return [ord(c) % 1000 for c in text[:500]]
    
    def _detokenize_single(self, token_id: int) -> str:
        """Detokenize 单个 token"""
        # TODO: 使用真实 tokenizer
        return chr(token_id + 65) if token_id < 97 else chr(token_id)
    
    def _generate_simple_response(self, input_text: str) -> str:
        """生成简单响应 (用于测试)"""
        responses = {
            "你好": "你好！我是类人脑AI 助手，基于Qwen3.5-0.8B 模型。我有什么可以帮助你的吗？",
            "介绍": "我是基于海马体 - 新皮层双系统架构的类人脑AI，具有 100Hz 高刷新推理、STDP 在线学习等特性。",
            "默认": "收到你的消息了！我正在使用类人脑架构进行处理，支持流式输出和实时交互。"
        }
        
        for key, value in responses.items():
            if key in input_text:
                return value
        
        return f"你说了：{input_text}。这是一个测试响应，实际使用时会连接真实的 Qwen3.5-0.8B 模型进行推理。"
    
    def set_callbacks(
        self,
        on_token: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """
        设置回调函数
        
        Args:
            on_token: token 生成回调
            on_complete: 完成回调
            on_error: 错误回调
        """
        self.on_token_callback = on_token
        self.on_complete_callback = on_complete
        self.on_error_callback = on_error


class TypingSimulator:
    """
    打字状态模拟器
    
    在 Telegram 中显示打字状态，提升用户体验
    """
    
    def __init__(self, bot, chat_id: int):
        """
        初始化打字模拟器
        
        Args:
            bot: Telegram Bot 实例
            chat_id: 聊天 ID
        """
        self.bot = bot
        self.chat_id = chat_id
        self.is_typing = False
    
    async def start_typing(self):
        """开始打字状态"""
        if not self.is_typing:
            try:
                await self.bot.send_chat_action(chat_id=self.chat_id, action='typing')
                self.is_typing = True
            except Exception as e:
                print(f"[Typing] 启动失败：{e}")
    
    async def stop_typing(self):
        """停止打字状态"""
        self.is_typing = False
    
    async def typing_context(self):
        """上下文管理器"""
        await self.start_typing()
        try:
            yield
        finally:
            await self.stop_typing()
