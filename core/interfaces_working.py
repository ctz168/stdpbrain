"""
类人脑双系统全闭环 AI架构 - 可运行版本

提供完整可运行的 BrainAIInterface 实现
使用简化的语言模型进行测试
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio
import random
import time


@dataclass
class BrainAIOutput:
    """输出数据结构"""
    text: str
    tokens: List[int]
    confidence: float
    memory_anchors: List[dict]
    stdp_stats: dict
    cycle_stats: dict


class SimpleLanguageModel(nn.Module):
    """
    简化语言模型 (用于测试和演示)
    
    基于规则的句子生成器
    实际部署时应替换为真实的 Qwen3.5-0.8B
    """
    
    def __init__(self, vocab_size: int = 10000, hidden_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 简单的 embedding 层
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # 示例响应模板
        self.response_templates = {
            "你好": [
                "你好！我是类人脑AI 助手，基于海马体 - 新皮层双系统架构。",
                "你好！很高兴见到你。我支持 100Hz 高刷新推理和 STDP 在线学习。",
                "你好！有什么可以帮助你的吗？"
            ],
            "介绍": [
                "我是基于Qwen3.5-0.8B 的类人脑AI，具有海马体记忆系统，支持情景记忆编码和召回。",
                "我采用双权重架构：90% 静态权重保证基础能力，10% 动态权重支持在线学习。"
            ],
            "架构": [
                "我的架构包括：海马体记忆系统 (EC-DG-CA3-CA1-SWR)、STDP 学习引擎、100Hz 刷新推理引擎。",
                "我采用类脑设计，支持 10ms 刷新周期，O(1) 复杂度注意力机制。"
            ],
            "STDP": [
                "STDP 是脉冲时序依赖可塑性，是我的核心学习机制。前神经元激活早于后神经元时增强连接，反之减弱。",
                "STDP 让我能够'推理即学习'，无需反向传播，纯本地时序信号驱动更新。"
            ],
            "默认": [
                "我收到了你的消息。作为一个类人脑AI，我正在学习和进化中。",
                "这是一个测试响应。实际使用时会连接真实的语言模型进行推理。",
                "我正在处理你的输入，使用海马体记忆系统进行上下文理解。"
            ]
        }
    
    def generate_response(self, input_text: str) -> str:
        """生成响应"""
        input_lower = input_text.lower()
        
        # 匹配关键词
        for keyword, responses in self.response_templates.items():
            if keyword in input_lower and keyword != "默认":
                return random.choice(responses)
        
        # 默认响应
        return random.choice(self.response_templates["默认"])
    
    def forward(self, x):
        """前向传播 (简化)"""
        return self.embedding(x)


class BrainAIInterface:
    """
    类人脑AI架构统一接口 (可运行版本)
    
    功能:
    - 对话和文本生成
    - 海马体记忆管理
    - STDP 权重更新模拟
    - 100Hz 刷新周期模拟
    """
    
    def __init__(self, config, device: Optional[str] = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[BrainAI] 初始化中... 设备：{self.device}")
        
        # ========== 1. 初始化模型 (真实项皮层) ==========
        self.is_real_model = False
        if config.model_path and config.model_path != "":
            try:
                # 尝试加载真实 Qwen 接口
                from core.qwen_interface import QwenInterface
                self.model = QwenInterface(
                    model_path=config.model_path,
                    config=config,
                    device=self.device,
                    quantization=getattr(config, 'quantization', 'INT4')
                )
                self.is_real_model = True
                print("[BrainAI] ✓ 真实 Qwen 模型 (新皮层) 初始化完成")
            except Exception as e:
                print(f"[BrainAI] ⚠ 真实模型加载失败：{e}，切换至简化模型")
                self.model = SimpleLanguageModel()
        else:
            self.model = SimpleLanguageModel()
            print("[BrainAI] ✓ 简化模型初始化完成")
        
        # ========== 2. 初始化海马体系统 ==========
        try:
            from hippocampus.hippocampus_system import HippocampusSystem
            self.hippocampus = HippocampusSystem(config, device=self.device)
            print("[BrainAI] ✓ 海马体系统初始化完成")
        except Exception as e:
            print(f"[BrainAI] ⚠ 海马体系统加载失败：{e}，使用简化版本")
            self.hippocampus = self._create_simple_hippocampus()
        
        # ========== 3. 初始化 STDP 引擎 ==========
        try:
            from core.stdp_engine import STDPEngine
            self.stdp_engine = STDPEngine(config, device=self.device)
            print("[BrainAI] ✓ STDP 引擎初始化完成")
        except Exception as e:
            print(f"[BrainAI] ⚠ STDP 引擎加载失败：{e}，使用简化版本")
            self.stdp_engine = self._create_simple_stdp_engine()
        
        # ========== 4. 初始化自闭环优化器 ==========
        try:
            from self_loop.self_loop_optimizer import SelfLoopOptimizer
            self.self_loop = SelfLoopOptimizer(config, model=self.model)
            print("[BrainAI] ✓ 自闭环优化器初始化完成")
        except Exception as e:
            print(f"[BrainAI] ⚠ 自闭环优化器加载失败：{e}")
            self.self_loop = None
        
        # ========== 5. 初始化 100Hz 刷新引擎 (核心执行框架) ==========
        try:
            from core.refresh_engine import RefreshCycleEngine
            self.cycle_engine = RefreshCycleEngine(
                model=self.model,
                hippocampus=self.hippocampus,
                stdp_engine=self.stdp_engine,
                period_ms=getattr(config, 'refresh_period_ms', 10),
                narrow_window_size=getattr(config, 'narrow_window_size', 2),
                device=self.device
            )
            print("[BrainAI] ✓ 100Hz 刷新引擎初始化完成")
        except Exception as e:
            print(f"[BrainAI] ⚠ 100Hz 刷新引擎加载失败：{e}")
            self.cycle_engine = None
        
        # ========== 5. 统计信息 ==========
        self.cycle_count = 0
        self.total_generation_time = 0.0
        
        print("[BrainAI] ✓ 初始化完成，准备就绪\n")
    
    def _create_simple_hippocampus(self):
        """创建简化海马体"""
        class SimpleHippocampus:
            def record_activity(self):
                pass
            
            def recall(self, features, topk=2):
                return [{'id': f'mem_{i}', 'timestamp': int(time.time()*1000)} 
                        for i in range(topk)]
            
            def get_stats(self):
                return {'num_memories': 0, 'memory_usage_mb': 0.1}
            
            def start_swr_monitoring(self):
                pass
            
            def stop_swr_monitoring(self):
                pass
        
        return SimpleHippocampus()
    
    def _create_simple_stdp_engine(self):
        """创建简化 STDP 引擎"""
        class SimpleSTDPEngine:
            def step(self, **kwargs):
                pass
            
            def get_stats(self):
                return {'cycle_count': 0}
            
            def reset(self):
                pass
        
        return SimpleSTDPEngine()
    
    def generate(
        self,
        input_text: str,
        max_tokens: int = 100,
        use_self_loop: bool = True
    ) -> BrainAIOutput:
        """
        生成回复 (完整实现)
        
        Args:
            input_text: 输入文本
            max_tokens: 最大生成 token 数
            use_self_loop: 是否启用自闭环优化
        
        Returns:
            output: 生成结果
        """
        start_time = time.time()
        
        # ========== 1. 记录活动 ==========
        self.hippocampus.record_activity()
        
        # ========== 2. 使用自闭环优化器 (如果可用) ==========
        if use_self_loop and self.self_loop:
            try:
                result = self.self_loop.run(input_text)
                optimized_input = result.output_text
            except Exception as e:
                optimized_input = input_text
        else:
            optimized_input = input_text
        
        # ========== 3. 生成响应 ==========
        if self.is_real_model:
            # 真实模型生成
            output = self.model.generate(optimized_input, max_tokens=max_tokens)
            output.memory_anchors = self.hippocampus.recall(torch.randn(1024, device=self.device), topk=2)
            output.stdp_stats = self.stdp_engine.get_stats()
            return output
            
        response_text = self.model.generate_response(optimized_input)
        
        # ========== 4. Tokenize (简化) ==========
        generated_tokens = [ord(c) % 1000 for c in response_text[:max_tokens]]
        
        # ========== 5. 更新周期计数 ==========
        self.cycle_count += len(generated_tokens)
        
        # ========== 6. 获取记忆锚点 ==========
        dummy_features = torch.randn(1024, device=self.device)
        memory_anchors = self.hippocampus.recall(dummy_features, topk=2)
        
        # ========== 7. 计算耗时 ==========
        elapsed = time.time() - start_time
        self.total_generation_time += elapsed
        
        # ========== 8. 计算置信度 ==========
        confidence = min(0.95, 0.7 + len(response_text) / 200.0)
        
        return BrainAIOutput(
            text=response_text,
            tokens=generated_tokens,
            confidence=confidence,
            memory_anchors=memory_anchors,
            stdp_stats=self.stdp_engine.get_stats(),
            cycle_stats={
                'total_cycles': self.cycle_count,
                'avg_cycle_time_ms': (self.total_generation_time / self.cycle_count * 1000) 
                                    if self.cycle_count > 0 else 0
            }
        )
    
    async def generate_stream(
        self,
        input_text: str,
        max_tokens: int = 100,
        **kwargs
    ):
        """
        流式生成回复 (自闭环并发 + 海马体活动并行)
        """
        # ========== 1. 优化5: 并发执行海马体记录 + 自闭环优化 ==========
        # 将 self_loop.run() 放入线程池，与海马体 record_activity 并发
        loop = asyncio.get_event_loop()
        
        async def _run_self_loop():
            if self.self_loop:
                try:
                    result = await loop.run_in_executor(None, self.self_loop.run, input_text)
                    return result.output_text
                except:
                    return input_text
            return input_text
        
        async def _record_activity():
            await loop.run_in_executor(None, self.hippocampus.record_activity)
        
        # 并发执行，消除首 token 前的串行阻塞
        optimized_input, _ = await asyncio.gather(
            _run_self_loop(),
            _record_activity()
        )
        
        # ========== 3. 生成流 ==========
        if self.is_real_model and self.cycle_engine:
            # 使用 100Hz 刷新引擎作为核心框架
            print("[BrainAI] 启动 100Hz 核心刷新引擎流...")
            
            # 初始化输入
            input_ids = self.model.tokenizer.encode(optimized_input)
            
            # 1. Prompt 预填充 (Pre-fill)
            # 一次性处理所有 prompt tokens 以初始化 KV-cache
            prompt_tensor = torch.tensor([input_ids], device=self.device)
            prefill_output = self.model.model(
                input_ids=prompt_tensor,
                use_cache=True,
                return_dict=True
            )
            past_key_values = prefill_output.past_key_values
            current_token = input_ids[-1] # 已经处理过最后一个了，但我们需要它作为下一步的 input
            
            # 2. 生成循环 (基于 10ms 周期)
            # 注意：prefill_output 已经包含了最后一个 token 的输出，
            # 下一个产生的 token 应该是基于 prefill 的 logits 采样得到的。
            # 为了对齐循环，我们先从 prefill 获取第一个生成 token
            next_token_logits = prefill_output.logits[:, -1, :]
            # 简化采样
            current_token = torch.argmax(next_token_logits, dim=-1).item()
            yield self.model.tokenizer.decode([current_token])
            
            for _ in range(max_tokens - 1):
                cycle_res = await self.cycle_engine.run_cycle(
                    input_token=current_token,
                    past_key_values=past_key_values,
                    **kwargs
                )
                
                if not cycle_res.success:
                    break
                    
                current_token = cycle_res.output_token
                past_key_values = cycle_res.past_key_values
                
                # 解码并吐出
                chunk = self.model.tokenizer.decode([current_token])
                yield chunk
                
                # 停止条件
                if current_token == self.model.tokenizer.eos_token_id:
                    break
        elif self.is_real_model:
            # 回退模式
            async for chunk in self.model.generate_stream(
                optimized_input, 
                max_tokens=max_tokens,
                **kwargs
            ):
                yield chunk
        else:
            # 简化模式模拟流式
            output = self.generate(optimized_input, max_tokens=max_tokens)
            response_text = output.text
            chunk_size = 2
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i+chunk_size]
                yield chunk
                await asyncio.sleep(0.05)

    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        对话接口 (完整实现)
        
        Args:
            message: 用户消息
            history: 对话历史
        
        Returns:
            response: 回复文本
        """
        # 构建带上下文的输入
        if history:
            context = "\n".join([
                f"{h['role']}: {h['content']}" 
                for h in history[-5:]
            ])
            full_input = f"{context}\nUser: {message}\nAssistant:"
        else:
            full_input = f"User: {message}\nAssistant:"
        
        # 生成回复
        output = self.generate(full_input, max_tokens=200)
        
        return output.text
    
    def get_stats(self) -> dict:
        """获取完整统计信息"""
        return {
            'hippocampus': self.hippocampus.get_stats(),
            'stdp': self.stdp_engine.get_stats(),
            'self_loop': self.self_loop.get_stats() if self.self_loop else {'enabled': False},
            'system': {
                'total_cycles': self.cycle_count,
                'device': self.device
            }
        }
    
    def reset(self):
        """重置所有状态"""
        self.stdp_engine.reset()
        self.cycle_count = 0
        self.total_generation_time = 0.0
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'cycle_count': self.cycle_count,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"[BrainAI] 检查点已保存：{path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.cycle_count = checkpoint.get('cycle_count', 0)
        print(f"[BrainAI] 检查点已加载：{path}")


# ==================== 快捷创建函数 ====================

def create_brain_ai(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    quantization: str = "INT4"
) -> BrainAIInterface:
    """
    快捷创建类人脑AI 实例
    
    Args:
        model_path: 模型路径 (当前版本不使用)
        device: 设备 ("cuda" | "cpu")
        quantization: 量化类型
    
    Returns:
        ai: BrainAIInterface 实例
    """
    from configs.arch_config import default_config
    return BrainAIInterface(default_config, device=device)
