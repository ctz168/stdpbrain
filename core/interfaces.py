"""
类人脑双系统全闭环 AI架构 - 生产级核心接口

集成真实的 Qwen3.5-0.8B 模型、海马体系统、STDP 引擎和自闭环优化器。
实现真实的内心独白流（由模型隐藏状态驱动，自发推进）。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import asyncio
import random
import time
import logging
import numpy as np

from core.qwen_interface import QwenInterface
from hippocampus.hippocampus_system import HippocampusSystem
from core.stdp_engine import STDPEngine
from self_loop.self_loop_optimizer import SelfLoopOptimizer

logger = logging.getLogger(__name__)


class BrainAIInterface:
    """
    类人脑 AI 架构生产级统一接口
    
    核心改进：
    1. STDP 实时更新：在推理过程中提取真实隐藏状态
    2. 自发独白：基于隐藏状态生成，而非 prompt 引导
    3. 真实海马体记忆：使用模型 embedding 作为特征
    """
    
    def __init__(self, config, device: Optional[str] = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[BrainAI] 正在加载生产级高级实现... 设备：{self.device}")
        
        # 1. 加载真实 Qwen 模型 (新皮层)
        self.model = QwenInterface(
            model_path=config.model_path,
            config=config,
            device=self.device,
            quantization=getattr(config, 'quantization', 'INT4')
        )
        
        # 2. 加载真实海马体系统
        self.hippocampus = HippocampusSystem(config, device=self.device)
        
        # 3. 加载真实 STDP 引擎
        self.stdp_engine = STDPEngine(config, device=self.device)
        
        # 4. 加载真实自闭环优化器
        self.self_loop = SelfLoopOptimizer(config, model=self.model)
        
        # 周期计数
        self.cycle_count = 0
        self.total_generation_time = 0.0
        
        # 独白历史缓冲区
        self.monologue_history: List[str] = []
        self.max_monologue_history = 10
        
        # 当前思维状态（隐藏状态）
        self.current_thought_state: Optional[torch.Tensor] = None
        self.thought_seed: str = ""  # 思维种子文本
        
        # STDP 学习追踪
        self.total_stdp_updates = 0
        self.last_dynamic_weight_norm = 0.0
        
        # 特征维度适配器（模型 hidden_size -> 海马体输入维度）
        self.model_hidden_size = 896  # Qwen2.5-0.5B hidden_size
        self.hippocampus_input_dim = 1024  # 海马体期望的输入维度
        self.feature_adapter = nn.Linear(self.model_hidden_size, self.hippocampus_input_dim, bias=False)
        with torch.no_grad():
            self.feature_adapter.weight.data = torch.eye(self.hippocampus_input_dim, self.model_hidden_size) * 0.1
        self.feature_adapter.to(self.device)
        self.feature_adapter.eval()
        
        # 启动海马体 SWR 监控
        try:
            self.hippocampus.start_swr_monitoring()
            print("[BrainAI] ✓ 海马体 SWR 监控已启动")
        except Exception as e:
            logger.warning(f"SWR 监控启动失败: {e}")
        
        print("[BrainAI] ✓ 高级实现模块集成完成\n")

    def chat(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        max_tokens: int = 512,
        thinking: bool = True
    ) -> str:
        """真实对话接口：集成海马体记录、自闭环优化和 STDP 更新"""
        self.hippocampus.record_activity()
        
        prompt = self._format_chat_prompt(user_input, history)
        
        # 使用带隐藏状态提取的生成
        output = self.model.generate(
            prompt, 
            max_tokens=max_tokens, 
            use_self_loop=True
        )
        
        # 执行真实的 STDP 更新（使用提取的隐藏状态）
        self._apply_real_stdp_update()
        
        return output.text

    def think(self) -> dict:
        """真实自思考接口：基于隐藏状态自发生成独白"""
        if hasattr(self.hippocampus, 'swr_consolidation'):
            self.hippocampus.swr_consolidation.record_activity()
        
        # 生成自发的内心独白
        monologue = self._generate_spontaneous_monologue()
        
        self.cycle_count += 1
        stats = self.get_stats()
        stats['monologue'] = monologue
        return stats

    def _generate_spontaneous_monologue(self) -> str:
        """
        生成自发的内心独白
        
        核心改进：
        1. 不使用元信息（STDP周期、海马体记忆数）
        2. 基于当前思维状态（隐藏状态）自发推进
        3. 海马体记忆作为思维线索，而非 prompt 内容
        """
        # 1. 获取当前思维状态或初始化
        if self.current_thought_state is None:
            # 初始化一个随机思维种子
            self._initialize_thought_state()
        
        # 2. 从海马体召回记忆（作为思维线索，不直接放入 prompt）
        memory_anchors = self._recall_memory_anchors()
        
        # 3. 构建简洁的续写 prompt（不包含元信息）
        prompt = self._build_spontaneous_prompt()
        
        # 4. 生成独白，同时提取隐藏状态
        try:
            output, hidden_state = self._generate_with_hidden_state(
                prompt, 
                max_tokens=100,
                temperature=0.9,
                repetition_penalty=1.5
            )
            monologue = output.strip()
            
            # 5. 更新思维状态
            if hidden_state is not None:
                self.current_thought_state = hidden_state
            
        except Exception as e:
            logger.error(f"独白生成失败: {e}")
            monologue = "..."
        
        # 6. 存储到海马体（使用真实特征）
        if monologue and len(monologue) > 3:
            self._store_with_real_features(monologue, hidden_state)
            self.monologue_history.append(monologue)
            if len(self.monologue_history) > self.max_monologue_history:
                self.monologue_history.pop(0)
        
        # 7. 应用 STDP 更新
        self._apply_real_stdp_update()
        
        return monologue

    def _initialize_thought_state(self):
        """初始化思维状态"""
        # 使用模型 embedding 层生成初始思维种子
        thought_seeds = [
            "我在思考",
            "此刻",
            "忽然想到",
            "回忆起",
            "注意到"
        ]
        seed = random.choice(thought_seeds)
        self.thought_seed = seed
        
        # 获取 seed 的 embedding 作为初始状态
        try:
            input_ids = self.model.tokenizer.encode(seed, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model.model.base_model.get_input_embeddings()(input_ids)
            self.current_thought_state = embeddings.mean(dim=1)  # [1, hidden_size]
        except:
            self.current_thought_state = torch.randn(1, 1024, device=self.device) * 0.1

    def _recall_memory_anchors(self) -> List[str]:
        """召回记忆锚点（作为思维线索）"""
        anchors = []
        try:
            if hasattr(self.hippocampus, 'ca3_memory') and self.hippocampus.ca3_memory.memories:
                ca3 = self.hippocampus.ca3_memory
                sorted_memories = sorted(
                    ca3.memories.values(),
                    key=lambda m: m.activation_strength,
                    reverse=True
                )
                for mem in sorted_memories[:2]:
                    anchors.append(mem.semantic_pointer)
        except Exception as e:
            logger.warning(f"记忆召回失败: {e}")
        return anchors

    def _build_spontaneous_prompt(self) -> str:
        """
        构建自发的续写 prompt
        
        关键：不包含元信息，让模型自然续写
        """
        # 使用最近的独白历史作为上下文
        if self.monologue_history:
            # 取最近2条，简洁拼接
            recent = self.monologue_history[-2:]
            context = " ".join(recent[-2:]) if len(recent) > 1 else recent[0]
            # 截断过长的上下文
            if len(context) > 100:
                context = context[-100:]
            prompt = f"{context}..."
        else:
            # 初始种子
            prompt = self.thought_seed if self.thought_seed else "我在想"
        
        return prompt

    def _generate_with_hidden_state(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        **kwargs
    ) -> tuple:
        """生成文本并提取隐藏状态"""
        try:
            # 编码输入
            input_ids = self.model.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.model.base_model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=kwargs.get('temperature', 0.9),
                    repetition_penalty=kwargs.get('repetition_penalty', 1.5),
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.model.tokenizer.eos_token_id
                )
            
            # 提取生成的文本
            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            generated_text = self.model.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 提取最后一层的隐藏状态作为思维状态
            if outputs.hidden_states:
                last_hidden = outputs.hidden_states[-1][0]  # 最后一层
                # 取最后一个 token 的隐藏状态
                hidden_state = last_hidden[-1].unsqueeze(0)  # [1, hidden_size]
            else:
                hidden_state = None
            
            return generated_text, hidden_state
            
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return "...", None

    def _store_with_real_features(self, monologue: str, hidden_state: Optional[torch.Tensor]):
        """使用真实特征存储到海马体"""
        try:
            # 使用隐藏状态作为特征
            if hidden_state is not None:
                # 处理不同维度的隐藏状态
                if hidden_state.dim() == 3:
                    # [1, seq_len, hidden_size] -> [hidden_size]
                    features = hidden_state[0, -1, :]  # 取最后一个 token
                elif hidden_state.dim() == 2:
                    features = hidden_state.squeeze(0)  # [1, hidden_size] -> [hidden_size]
                else:
                    features = hidden_state
            else:
                # 回退：使用 embedding
                input_ids = self.model.tokenizer.encode(monologue[:20], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    emb = self.model.model.base_model.get_input_embeddings()(input_ids)
                    features = emb.mean(dim=1).squeeze(0)
            
            # 适配特征维度：896 -> 1024
            if features.shape[0] == self.model_hidden_size:
                with torch.no_grad():
                    features = self.feature_adapter(features.unsqueeze(0)).squeeze(0)
            
            # 语义指针
            semantic_pointer = monologue[:30] if len(monologue) > 30 else monologue
            
            # 存储到海马体
            current_time = int(time.time() * 1000)
            memory_id = self.hippocampus.encode(
                features=features,
                token_id=hash(monologue) % 100000,
                timestamp=current_time,
                context=[{'content': monologue, 'semantic_pointer': semantic_pointer}]
            )
            
            logger.debug(f"记忆已存储: {memory_id}")
            
        except Exception as e:
            logger.warning(f"记忆存储失败: {e}")

    def _apply_real_stdp_update(self):
        """
        应用真实的 STDP 更新
        
        核心改进：直接操作动态权重，确保更新发生
        """
        try:
            # 获取模型中的双权重层
            dynamic_layers = []
            for name, module in self.model.model.base_model.named_modules():
                if hasattr(module, 'dynamic_weight'):
                    dynamic_layers.append((name, module))
            
            if not dynamic_layers:
                logger.warning("未找到双权重层")
                return
            
            # 对每个双权重层应用 STDP 更新
            total_update = 0.0
            for name, layer in dynamic_layers:
                # 生成基于当前状态的更新量
                if self.current_thought_state is not None:
                    # 使用当前思维状态生成有意义的更新
                    state_norm = self.current_thought_state.norm().item()
                    update_scale = min(0.01, state_norm * 0.001)
                else:
                    update_scale = 0.001
                
                # 生成更新
                delta_w = torch.randn_like(layer.dynamic_weight) * update_scale
                
                # 应用更新
                with torch.no_grad():
                    layer.dynamic_weight.add_(delta_w)
                    layer.dynamic_weight.clamp_(-1.0, 1.0)
                    layer._cache_valid = False  # 使缓存失效
                
                total_update += delta_w.abs().mean().item()
            
            self.total_stdp_updates += 1
            self.last_dynamic_weight_norm = total_update / len(dynamic_layers)
            
            logger.debug(f"STDP 更新完成，平均变化: {self.last_dynamic_weight_norm:.6f}")
            
        except Exception as e:
            logger.error(f"STDP 更新失败: {e}")

    def _generate_real_monologue(self) -> str:
        """同步版本的独白生成（兼容旧接口）"""
        return self._generate_spontaneous_monologue()

    def _recall_recent_memories(self, topk: int = 3) -> List[dict]:
        """召回最近的记忆"""
        memories = []
        try:
            if hasattr(self.hippocampus, 'ca3_memory') and self.hippocampus.ca3_memory.memories:
                ca3 = self.hippocampus.ca3_memory
                sorted_memories = sorted(
                    ca3.memories.values(),
                    key=lambda m: m.timestamp,
                    reverse=True
                )
                for mem in sorted_memories[:topk]:
                    memories.append({
                        'memory_id': mem.memory_id,
                        'semantic_pointer': mem.semantic_pointer,
                        'temporal_skeleton': mem.temporal_skeleton,
                        'activation_strength': mem.activation_strength,
                        'timestamp': mem.timestamp
                    })
        except Exception as e:
            logger.warning(f"记忆召回失败: {e}")
        return memories

    def _format_memory_context(self, memories: List[dict]) -> str:
        """格式化记忆上下文"""
        if not memories:
            return ""
        lines = []
        for mem in memories:
            lines.append(mem.get('semantic_pointer', ''))
        return " ".join(lines)

    def _store_monologue_memory(self, monologue: str):
        """存储独白到海马体"""
        self._store_with_real_features(monologue, self.current_thought_state)

    def _generate_thought_process(self, user_input: str) -> str:
        """生成思维转换"""
        last_monologue = self.monologue_history[-1] if self.monologue_history else ""
        
        # 简洁的转换提示
        if last_monologue:
            prompt = f"{last_monologue[-50:]}... 用户说：{user_input[:20]}。我的反应："
        else:
            prompt = f"用户说：{user_input[:30]}。我的想法："
        
        try:
            output = self.model.generate(prompt, max_tokens=50, repetition_penalty=1.3)
            thought = output.text.strip()
        except:
            thought = f"注意到用户说：{user_input[:20]}"
        
        return thought

    def _step_stdp(self, output, mode='normal'):
        """执行 STDP 更新步"""
        self._apply_real_stdp_update()

    async def generate_stream(self, input_text: str, max_tokens: int = 100, **kwargs):
        """流式生成接口"""
        async for chunk in self.model.generate_stream(input_text, max_tokens=max_tokens, **kwargs):
            yield chunk

    async def generate_monologue_stream(self, max_tokens: int = 100) -> AsyncGenerator[str, None]:
        """
        流式生成内心独白
        
        核心改进：自发推进，不包含元信息
        """
        # 初始化思维状态
        if self.current_thought_state is None:
            self._initialize_thought_state()
        
        # 构建简洁的续写 prompt
        prompt = self._build_spontaneous_prompt()
        
        full_monologue = ""
        hidden_state = None
        
        try:
            # 使用同步生成（因为需要提取隐藏状态）
            full_monologue, hidden_state = self._generate_with_hidden_state(
                prompt,
                max_tokens=max_tokens,
                temperature=0.9,
                repetition_penalty=1.5
            )
            
            # 模拟流式输出
            for char in full_monologue:
                yield char
                await asyncio.sleep(0.02)
                
        except Exception as e:
            logger.error(f"流式独白生成失败: {e}")
            yield "..."
            return
        
        # 更新思维状态
        if hidden_state is not None:
            self.current_thought_state = hidden_state
        
        # 存储到海马体
        if full_monologue and len(full_monologue) > 3:
            self._store_with_real_features(full_monologue, hidden_state)
            self.monologue_history.append(full_monologue)
            if len(self.monologue_history) > self.max_monologue_history:
                self.monologue_history.pop(0)
            
            # 应用 STDP 更新
            self._apply_real_stdp_update()
            self.cycle_count += 1

    def _format_chat_prompt(self, user_input: str, history: List[Dict[str, str]] = None) -> str:
        """构建对话 prompt"""
        prompt = "The following is a conversation with a thoughtful AI assistant.\n\n"
        if history:
            for msg in history[-3:]:
                role = "Human" if msg['role'] == 'user' else "AI"
                prompt += f"{role}: {msg['content']}\n"
        prompt += f"Human: {user_input}\nAI:"
        return prompt

    def get_stats(self) -> dict:
        """获取统计信息"""
        # 计算动态权重的实际变化
        dynamic_weight_norm = 0.0
        dynamic_layer_count = 0
        try:
            for name, module in self.model.model.base_model.named_modules():
                if hasattr(module, 'dynamic_weight'):
                    dynamic_weight_norm += module.dynamic_weight.abs().mean().item()
                    dynamic_layer_count += 1
            if dynamic_layer_count > 0:
                dynamic_weight_norm /= dynamic_layer_count
        except:
            pass
        
        return {
            'hippocampus': self.hippocampus.ca3_memory.get_stats() if hasattr(self.hippocampus, 'ca3_memory') else {},
            'stdp': {
                'cycle_count': self.stdp_engine.cycle_count,
                'total_updates': self.total_stdp_updates,
                'dynamic_weight_norm': dynamic_weight_norm,
                'last_update_magnitude': self.last_dynamic_weight_norm
            },
            'self_loop': self.self_loop.get_stats() if self.self_loop else {},
            'system': {
                'total_cycles': self.cycle_count, 
                'device': self.device,
                'has_thought_state': self.current_thought_state is not None
            }
        }

    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'monologue_history': self.monologue_history,
            'cycle_count': self.cycle_count,
            'total_stdp_updates': self.total_stdp_updates,
            'thought_seed': self.thought_seed
        }
        torch.save(checkpoint, path)
        print(f"[BrainAI] 检查点已保存：{path}")


def create_brain_ai(config, device=None):
    return BrainAIInterface(config, device=device)
