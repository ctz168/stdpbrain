"""Telegram Bot 流式输出处理器

核心功能:
- 实时流式生成响应
- Token-by-token 输出
- 打字状态模拟
- 真实 Qwen 模型集成
"""

import asyncio
import time
from typing import Callable, Optional, List, Dict, Any


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
       model=None,
       tokenizer=None,
      delay_ms: int = 50,
       chunk_size: int = 1,
      max_tokens: int = 500
    ):
        """
       初始化流式处理器
        
       Args:
         model: Qwen 模型 (可选)
           tokenizer: Qwen tokenizer (可选)
         delay_ms: token 间延迟 (毫秒)
         chunk_size: 每次输出的 chunk 大小
         max_tokens: 最大生成 token 数
        """
    self.model = model
   self.tokenizer= tokenizer
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
      context: Optional[str] = None
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
      
     try:
        self.total_streams += 1
          
         # ==========1. 使用自闭环优化 (如果已配置) ==========
       if context and hasattr(self, 'self_loop_optimizer'):
          from .self_loop_optimizer import SelfLoopOptimizer
           optimizer = SelfLoopOptimizer(config=None, model={'model': self.model, 'tokenizer': self.tokenizer})
          result= optimizer.run(input_text, context=[context])
           optimized_input = result.output_text
        else:
          optimized_input = input_text
          
         # ==========2. 使用真实模型流式生成 ==========
       if self.model is not None and self.tokenizer is not None:
          print(f"[Stream] 使用真实 Qwen 模型流式生成...")
            
            # Tokenize 输入
          input_tokens = self._tokenize(optimized_input)
            
            # 导入 torch
          import torch
                
            # 准备输入
          inputs = self.tokenizer(optimized_input, return_tensors="pt")
         input_length = inputs.input_ids.shape[1]
            
            # 自回归生成
           generated_ids = []
          with torch.no_grad():
              for i in range(self.max_tokens):
                  # 前向传播
                 outputs = self.model.generate(
                     **inputs,
                    max_new_tokens=1,
                     temperature=0.7,
                     do_sample=True,
                     top_p=0.9,
                   pad_token_id=self.tokenizer.eos_token_id,
                   use_cache=True
                 )
                  
                 # 获取新生成的 token
               new_token_id = outputs[0, -1].item()
                generated_ids.append(new_token_id)
                  
                 # Detokenize
                token_text = self.tokenizer.decode([new_token_id], skip_special_tokens=True)
                  
                 # 添加延迟模拟打字效果
               await asyncio.sleep(self.delay_ms / 1000.0)
                  
                 # 输出 token
               yield token_text
               full_response += token_text
                  
              self.total_tokens_generated += 1
                  
               # 检查是否生成结束符
             if new_token_id == self.tokenizer.eos_token_id:
                   break
                  
               # 更新输入 (包含已生成的 token)
             inputs = torch.cat([
                  inputs.input_ids,
                   torch.tensor([[new_token_id]])
               ], dim=1)
            
            # 如果没有生成内容，使用简化响应
          if not full_response:
              full_response = self._generate_simple_response(optimized_input)
               # 分块输出简化响应
              chunk_size = 20
              for i in range(0, len(full_response), chunk_size):
                   chunk = full_response[i:i+chunk_size]
                  yield chunk
                  await asyncio.sleep(self.delay_ms / 1000.0)
        else:
           # ==========3. 降级：使用简化实现 ==========
          print(f"[Stream] 使用简化实现 (未加载模型)...")
          full_response = self._generate_simple_response(optimized_input)
            
           # 按字符分块输出
          chunk_size = 3
          for i in range(0, len(full_response), chunk_size):
              chunk = full_response[i:i+chunk_size]
             yield chunk
             await asyncio.sleep(self.delay_ms / 1000.0)
          
         # ==========4. 完成回调 ==========
       if self.on_complete_callback:
          self.on_complete_callback(full_response)
        
        elapsed = time.time() - start_time
      self.avg_stream_length = (
           (self.avg_stream_length * (self.total_streams- 1) + len(full_response))
           / self.total_streams
       )
      print(f"[Stream] 生成完成，耗时：{elapsed*1000:.1f}ms, 长度：{len(full_response)}")
        
    except Exception as e:
      print(f"[Stream] 错误：{e}")
      if self.on_error_callback:
         self.on_error_callback(e)
      yield f"[错误] 生成失败：{str(e)}"
    
  def _tokenize(self, text: str) -> List[int]:
        """Tokenize 输入"""
     if self.tokenizer is not None:
         # 使用真实 tokenizer
        encoded = self.tokenizer.encode(text, return_tensors="pt")
       return encoded[0].tolist()
     else:
        # 降级：简单字符编码
      return [ord(c) for c in text[:500]]
    
  def _detokenize_single(self, token_id: int) -> str:
        """Detokenize 单个 token"""
     if self.tokenizer is not None:
         # 使用真实 tokenizer
      return self.tokenizer.decode([token_id], skip_special_tokens=True)
     else:
        # 降级：简单字符解码
      return chr(token_id % 128) if token_id < 128 else "?"
    
  def _generate_simple_response(self, input_text: str) -> str:
        """生成简单响应 (无模型时使用)"""
      responses = {
           "你好": "你好！我是类人脑 AI 助手，基于 Qwen3.5-0.8B 模型。我有什么可以帮助你的吗？",
           "介绍": "我是基于海马体 - 新皮层双系统架构的类人脑 AI，具有 100Hz 高刷新推理、STDP 在线学习等特性。",
           "默认": "收到你的消息了！我正在使用类人脑架构进行处理，支持流式输出和实时交互。"
       }
        
       for key, value in responses.items():
         if key in input_text:
            return value
        
     return f"你说了：{input_text}。这是一个测试响应，实际使用时会连接真实的 Qwen3.5-0.8B 模型进行推理。"
    
  def set_model(self, model, tokenizer):
        """设置模型和 tokenizer"""
     self.model = model
    self.tokenizer= tokenizer
    print(f"[StreamHandler] 模型已设置")
    
  def get_stats(self) -> dict:
        """获取统计信息"""
     return {
           'total_streams': self.total_streams,
           'total_tokens_generated': self.total_tokens_generated,
           'avg_stream_length': self.avg_stream_length
       }


class TypingSimulator:
    """打字模拟器"""
    
  def __init__(self, delay_ms: int = 100):
     self.delay_ms = delay_ms
    
  async def simulate_typing(self, text: str) -> AsyncGenerator[str, None]:
        """模拟打字效果"""
      # 按字符输出
     for char in text:
        yield char
       await asyncio.sleep(self.delay_ms / 1000.0)


if __name__ == "__main__":
    # 测试流式处理器
  async def test_stream():
     handler= StreamHandler()
      
    print("=" * 60)
    print("流式处理器测试")
    print("=" * 60)
      
     test_inputs = ["你好", "介绍一下你自己", "今天天气不错"]
      
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
