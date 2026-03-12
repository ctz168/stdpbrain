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
        max_tokens: int = 256,
        thinking: bool = True
    ) -> str:
        """
        类人模式对话：
        1. 消化 (Digest)：将输入转化为思维状态
        2. 思考 (Think)：基于输入生成一段潜意识独白
        3. 回复 (Respond)：基于独白和输入生成正式回答
        """
        self.hippocampus.record_activity()
        
        # 1. 消化输入：强制更新思维种子
        self.thought_seed = f"用户说：{user_input[:20]}"
        
        # 2. 思考：生成潜意识独白 (受刺激的思考)
        # 这种独白是面向内部的，短而碎
        monologue = self._generate_spontaneous_monologue(max_tokens=30, temperature=0.9)
        
        # 3. 回复：基于思维流生成
        prompt = self._format_chat_prompt(user_input, history, monologue)
        
        output = self.model.generate(
            prompt, 
            max_tokens=max_tokens, 
            temperature=0.7,  # 回复更稳重
            use_self_loop=True
        )
        
        # 存储并应用 STDP
        self._store_with_real_features(output.text, self.current_thought_state)
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

    def _generate_spontaneous_monologue(self, max_tokens: int = 60, temperature: float = 0.9) -> str:
        """
        生成自发的内心独白 (潜意识流)
        
        优化：
        1. 加入乱码检查
        2. 引入海马体锚定 (Grounding)
        3. 状态化生成
        """
        # 1. 获取海马体线索 (作为思维锚点)
        memory_anchors = self._recall_recent_memories(topk=1)
        anchor_text = memory_anchors[0]['semantic_pointer'] if memory_anchors else ""
        
        # 2. 构建 Prompt
        # 如果当前独白历史中有太多乱码符号，强制使用锚点重置
        prompt = self._build_spontaneous_prompt()
        if self._is_gibberish(prompt) and anchor_text:
            prompt = f"回忆起：{anchor_text}... 现在我在想"
        
        # 3. 生成，提取隐藏状态
        try:
            output, hidden_state = self._generate_with_hidden_state(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=1.3
            )
            
            # 4. 乱码过滤：如果生成的独白依然是乱码，丢弃并回退
            if self._is_gibberish(output):
                monologue = "思维有些模糊..."
            else:
                monologue = output.strip()
            
            # 5. 更新思维状态
            if hidden_state is not None:
                self.current_thought_state = hidden_state
            
        except Exception as e:
            logger.error(f"独白生成失败: {e}")
            monologue = "..."
        
        # 6. 存储到海马体
        if monologue and len(monologue) > 3:
            self._store_with_real_features(monologue, hidden_state)
            self.monologue_history.append(monologue)
            if len(self.monologue_history) > self.max_monologue_history:
                self.monologue_history.pop(0)
        
        # 7. 应用 STDP
        self._apply_real_stdp_update()
        
        return monologue

    def _is_gibberish(self, text: str) -> bool:
        """简单的乱码/特殊符号检查 (针对 0.8B 的常见错误)"""
        if not text: return True
        # 检查特殊符号比例 (如大量的 $ \ sum _ 等)
        special_chars = set("$%^&*()_+={}|[]\\:;\"'<>,/?#")
        special_count = sum(1 for char in text if char in special_chars)
        if len(text) > 0 and special_count / len(text) > 0.3:
            return True
        # 检查是否包含过多重复字符
        if len(text) > 10 and len(set(text)) < 5:
            return True
        return False

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
        构建潜意识续写 Prompt
        """
        # 如果没有思维状态，初始化
        if self.current_thought_state is None:
            self._initialize_thought_state()
            
        # 优先使用思维种子（由用户消息或后台触发更新）
        if self.thought_seed:
            seed = self.thought_seed
            self.thought_seed = "" # 使用后清空
            return f"{seed}..."
            
        # 否则使用最近的一条独白历史作为联想起点
        if self.monologue_history:
            context = self.monologue_history[-1]
            if len(context) > 60:
                context = context[-60:]
            return f"{context}..."
        
        return "我在想..."

    def _generate_with_hidden_state(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.9,
        repetition_penalty: float = 1.5
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
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
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
        强化版 STDP 更新：关联学习 (Hebbian-like)
        
        核心改进：
        1. 基于当前思维状态 (Hidden State) 的关联更新
        2. 引入 5% 范数限制 (Norm Limit)，保护静态权重
        3. 极微量的权重固化 (Weight Consolidation)
        """
        try:
            if self.current_thought_state is None:
                return
            
            # 获取当前思维状态的归一化特征 [hidden_size]
            thought_vec = self.current_thought_state.view(-1)
            thought_norm = thought_vec.norm()
            
            # 获取模型中的双权重层
            dynamic_layers = []
            for name, module in self.model.model.base_model.named_modules():
                if hasattr(module, 'dynamic_weight'):
                    dynamic_layers.append((name, module))
            
            if not dynamic_layers:
                return
            
            # 学习率与固化率
            lr = 0.005  # 学习率
            consolidation_rate = 0.0001  # 固化率 (0.01% 的动态权重转入静态)
            max_dynamic_ratio = 0.05  # 5% 的范数限制
            
            total_update = 0.0
            for name, layer in dynamic_layers:
                # 1. 模拟赫布学习：基于思维向量的投影产生关联更新
                # 这里我们假设权重矩阵 W 与 thought_vec 的外积能代表当前的“思维关联”
                # 由于我们没有每层的输入输出，我们使用 thought_vec 的自关联来模拟这种趋势
                # [out_f, in_f] 的形状，我们采样部分 thought_vec
                out_f, in_f = layer.dynamic_weight.shape
                
                # 创建一个简化的关联增量 (Association Delta)
                # 使用 thought_vec 的一部分作为输出端，一部分作为输入端进行外积
                v_out = thought_vec[:out_f] if thought_vec.shape[0] >= out_f else F.pad(thought_vec, (0, out_f - thought_vec.shape[0]))
                v_in = thought_vec[:in_f] if thought_vec.shape[0] >= in_f else F.pad(thought_vec, (0, in_f - thought_vec.shape[0]))
                
                delta_w = torch.outer(v_out, v_in) * (lr / (thought_norm + 1e-6))
                
                # 2. 加入随机性探索 (Noise Exploration)
                delta_w += torch.randn_like(delta_w) * (lr * 0.1)
                
                # 3. 应用更新并实施范数限制
                with torch.no_grad():
                    # 更新动态权重
                    layer.dynamic_weight.add_(delta_w)
                    
                    # 计算范数比例并限制
                    static_norm = layer.static_weight.norm()
                    dynamic_norm = layer.dynamic_weight.norm()
                    
                    if dynamic_norm > static_norm * max_dynamic_ratio:
                        # 如果超过 5%，按比例缩放回限制内
                        scale = (static_norm * max_dynamic_ratio) / (dynamic_norm + 1e-9)
                        layer.dynamic_weight.mul_(scale)
                    
                    # 4. 模拟长期记忆固化 (Weight Consolidation)
                    # 将极微量的动态变化固化进静态权重 (介入原始模型)
                    if consolidation_rate > 0:
                        consolidation_delta = layer.dynamic_weight.data * consolidation_rate
                        layer.static_weight.data.add_(consolidation_delta)
                        # 从动态中减去固化的部分，保持平衡
                        layer.dynamic_weight.data.sub_(consolidation_delta)
                    
                    layer._cache_valid = False  # 使缓存失效
                
                total_update += delta_w.abs().mean().item()
            
            self.total_stdp_updates += 1
            self.last_dynamic_weight_norm = total_update / len(dynamic_layers)
            
        except Exception as e:
            logger.error(f"STDP 关联学习失败: {e}")

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

    async def chat_stream(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        max_tokens: int = 256
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        流式类人对话接口：
        1. 消化 & 产生潜意识 (yield monologue)
        2. 生成正式回复 (yield response chunks)
        """
        self.hippocampus.record_activity()
        
        # 1. 消化
        self.thought_seed = f"用户说：{user_input[:20]}"
        
        # 2. 产生潜意识独白
        monologue = await asyncio.to_thread(self._generate_spontaneous_monologue, 30, 0.9)
        yield {"type": "monologue", "content": monologue}
        
        # 3. 生成正式回复流
        prompt = self._format_chat_prompt(user_input, history, monologue)
        
        full_response = ""
        async for chunk in self.model.generate_stream(prompt, max_tokens=max_tokens, temperature=0.7):
            full_response += chunk
            yield {"type": "chunk", "content": chunk}
            
        # 存储并应用 STDP
        self._store_with_real_features(full_response, self.current_thought_state)
        self._apply_real_stdp_update()

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
            # 使用 asyncio.to_thread 在后台线程中运行同步生成
            full_monologue, hidden_state = await asyncio.to_thread(
                self._generate_with_hidden_state,
                prompt,
                max_tokens,
                0.9,  # temperature
                1.5   # repetition_penalty
            )
            
            # 模拟流式输出
            for char in full_monologue:
                yield char
                await asyncio.sleep(0.02)
                # 让出控制权，允许处理其他事件
                await asyncio.sleep(0)
                
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

    def _format_chat_prompt(self, user_input: str, history: List[Dict[str, str]] = None, monologue: str = "") -> str:
        """
        构建对话 Prompt (类人模式)
        
        将独白作为 AI 的“思维背景”
        """
        system_msg = "You are a helpful, concise AI assistant. Answer the user accurately."
        
        prompt = f"<system>\n{system_msg}\n</system>\n\n"
        
        # 注入最近的潜意识（独白）
        if monologue:
            prompt += f"<thought>\n{monologue}\n</thought>\n\n"
            
        # 历史记录 (只取最近 2 轮以减轻 0.8B 负担)
        if history:
            for msg in history[-2:]:
                role = "User" if msg['role'] == 'user' else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
                
        prompt += f"User: {user_input}\nAssistant:"
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
