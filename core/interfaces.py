"""
类人脑双系统全闭环 AI架构 - 生产级核心接口

集成真实的 Qwen3.5-0.8B 模型、海马体系统、STDP 引擎和自闭环优化器。
实现真实的内心独白流（由模型隐藏状态驱动，自发推进）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from core.monologue_engine import MonologueEngine, ThoughtState, EmotionState

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
        
        # 5. 加载类人脑独白引擎
        self.monologue_engine = None  # 延迟初始化，需要等模型加载完成
        
        # 周期计数
        self.cycle_count = 0
        self.total_generation_time = 0.0
        
        # 独白历史缓冲区
        self.monologue_history: List[str] = []
        self.max_monologue_history = 10
        
        # 当前思维状态（隐藏状态）
        self.current_thought_state: Optional[torch.Tensor] = None
        self.thought_seed: str = ""  # 思维种子文本
        
        # 思维状态机状态（用于兼容旧接口）
        self._internal_thought_state = ThoughtState.RESTING
        self._internal_emotion_state = EmotionState.CALM
        
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
        
        # 状态文件路径
        self.state_path = "brain_state.pt"
        
        # 尝试加载现有状态，如果失败则执行创世注入
        if not self.load_state(self.state_path):
            self._seed_genesis_memory()
        
        # 注入唤醒记忆
        self._inject_wakeup_memory()
        
        # 初始化独白引擎
        try:
            self.monologue_engine = MonologueEngine(
                model_interface=self.model,
                hippocampus_system=self.hippocampus,
                config=config,
                device=self.device
            )
            print("[BrainAI] ✓ 类人脑独白引擎已初始化")
        except Exception as e:
            logger.warning(f"独白引擎初始化失败: {e}，将使用简化版本")
        
        # 启动海马体 SWR 监控
        try:
            self.hippocampus.start_swr_monitoring()
            print("[BrainAI] ✓ 海马体 SWR 监控已启动")
        except Exception as e:
            logger.warning(f"SWR 监控启动失败: {e}")
        
        # 设置海马体门控函数（连接CA1到注意力层）
        self._setup_hippocampus_gate()
        
        print("[BrainAI] ✓ 高级实现模块集成完成\n")
    
    def _setup_hippocampus_gate(self):
        """设置海马体门控，让CA1门控信号影响注意力"""
        def hippocampus_gate_fn(query, key, memory_anchors):
            """
            海马体门控函数
            
            Args:
                query: [batch, heads, seq_len, head_dim]
                key: [batch, heads, seq_len, head_dim]
                memory_anchors: 记忆锚点列表
            
            Returns:
                gate_mask: 注意力偏置 [batch, heads, seq_len, seq_len]
            """
            if not memory_anchors:
                return None
            
            batch_size, num_heads, seq_len, head_dim = query.shape
            
            # 使用CA1门控生成注意力偏置
            try:
                gate_signal = self.hippocampus.ca1_gate(
                    query.transpose(1, 2).reshape(-1, seq_len, num_heads * head_dim),
                    key.transpose(1, 2).reshape(-1, seq_len, num_heads * head_dim),
                    memory_anchors
                )
                # gate_signal: [batch, 1, seq_len, hidden_size]
                # 转换为注意力偏置
                if gate_signal is not None:
                    # 简化：使用门控信号的平均值作为偏置
                    bias = gate_signal.mean(dim=-1, keepdim=True)  # [batch, 1, seq_len, 1]
                    bias = bias.expand(-1, num_heads, -1, seq_len)  # [batch, heads, seq_len, seq_len]
                    return bias * 0.1  # 缩放因子
            except Exception as e:
                logger.debug(f"海马体门控计算失败: {e}")
            
            return None
        
        # 设置到模型的所有注意力层
        try:
            self.model.model.set_hippocampus_gate(hippocampus_gate_fn)
            print("[BrainAI] ✓ 海马体门控已连接到注意力层")
        except Exception as e:
            logger.warning(f"设置海马体门控失败: {e}")

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
        2. 召回 (Recall)：从海马体召回相关记忆
        3. 思考 (Think)：基于输入生成一段潜意识独白
        4. 回复 (Respond)：基于记忆、独白和输入生成正式回答
        5. 学习 (Learn)：STDP更新和记忆存储
        """
        self.hippocampus.record_activity()
        
        # 1. 消化输入：强制更新思维种子
        self.thought_seed = f"用户说：{user_input[:20]}"
        
        # 2. 召回海马体记忆（新增）
        memory_context = ""
        recalled_memories = []  # 保存召回的记忆锚点
        try:
            # 使用输入文本的embedding作为查询
            input_ids = self.model.tokenizer.encode(user_input[:50], return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model.model.base_model.get_input_embeddings()(input_ids)
            query_features = embeddings.mean(dim=1).squeeze(0)
            
            # 适配维度
            if query_features.shape[0] != 1024:
                query_features = self.feature_adapter(query_features.unsqueeze(0)).squeeze(0)
            
            # 召回记忆
            recalled_memories = self.hippocampus.recall(query_features, topk=2)
            if recalled_memories:
                memory_pointers = [m['semantic_pointer'] for m in recalled_memories if m.get('semantic_pointer')]
                if memory_pointers:
                    memory_context = "相关记忆: " + " | ".join(memory_pointers[:2])
        except Exception as e:
            logger.debug(f"记忆召回失败: {e}")
        
        # 3. 思考：生成潜意识独白 (受刺激的思考)
        # 设置思维种子，让独白响应用户输入
        self.thought_seed = f"用户说：{user_input[:30]}"
        monologue = self._generate_spontaneous_monologue(max_tokens=40, temperature=0.7)
        
        # 4. 回复：基于记忆和思维流生成
        prompt = self._format_chat_prompt(user_input, history, monologue, memory_context)
        
        # 准备记忆锚点张量（用于门控）
        memory_anchor = None
        if recalled_memories:
            try:
                # 提取记忆特征向量
                mem_features = []
                for mem in recalled_memories[:2]:
                    if 'dg_features' in mem and mem['dg_features'] is not None:
                        mem_features.append(mem['dg_features'])
                if mem_features:
                    memory_anchor = torch.stack(mem_features).mean(dim=0).unsqueeze(0).to(self.device)
            except Exception as e:
                logger.debug(f"准备记忆锚点失败: {e}")
        
        output = self.model.generate(
            prompt, 
            max_tokens=max_tokens, 
            temperature=0.7,  # 回复更稳重
            use_self_loop=True,
            memory_anchor=memory_anchor  # 传入记忆锚点
        )
        
        # 5. 存储用户输入和模型回复到海马体（保存完整上下文）
        # 提取关键信息作为语义指针
        semantic_pointer = f"用户: {user_input[:50]} | 回复: {output.text[:50]}"
        self._store_with_real_features(
            f"{user_input} -> {output.text}", 
            self.current_thought_state,
            semantic_pointer=semantic_pointer
        )
        self._apply_real_stdp_update()
        
        # 6. 调用STDP引擎的step方法（新增）
        try:
            self.stdp_engine.step(
                model_components={'hippocampus': self.hippocampus},
                inputs={
                    'context_tokens': torch.tensor([1, 2, 3]),
                    'current_token': hash(user_input) % 10000,
                    'memory_anchor_id': f'mem_{hash(output.text) % 10000}'
                },
                outputs={
                    'evaluation_score': 30 + len(output.text) % 20
                }
            )
        except Exception as e:
            logger.debug(f"STDP step失败: {e}")
        
        # 清理输出中的标签泄露
        cleaned_text = self._clean_output(output.text)
        return cleaned_text

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

    def _generate_spontaneous_monologue(self, max_tokens: int = 30, temperature: float = 0.6) -> str:
        """
        生成自发的内心独白 (潜意识流)
        
        优化：
        1. 使用ChatML格式
        2. 降低温度提高连贯性
        3. 简洁的独白内容
        4. 乱码检查和过滤
        """
        # 1. 构建Prompt (已使用ChatML格式)
        prompt = self._build_spontaneous_prompt()
        
        # 2. 生成独白
        try:
            output, hidden_state = self._generate_with_hidden_state(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=1.1
            )
            
            # 4. 乱码过滤
            if self._is_gibberish(output):
                monologue = "思维有些模糊..."
            else:
                monologue = output.strip()
            
            # 移除可能的格式残留
            for tag in ['<|im_end|>', '<|im_start|>', '</system>', '<system>', '</user>', '<user>']:
                monologue = monologue.replace(tag, '')
            
            monologue = monologue.strip()
            
            # 如果是乱码，使用默认值
            if self._is_gibberish(monologue) or len(monologue) < 2:
                monologue = "思考中..."
            
            # 截断到合理长度
            if len(monologue) > 50:
                monologue = monologue[:50] + "..."
            
            # 4. 更新思维状态
            if hidden_state is not None:
                self.current_thought_state = hidden_state
            
        except Exception as e:
            logger.error(f"独白生成失败: {e}")
            monologue = "思考中..."
        
        # 5. 存储到海马体
        if monologue and len(monologue) > 3:
            semantic_pointer = f"思考: {monologue[:30]}"
            self._store_with_real_features(monologue, hidden_state, semantic_pointer=semantic_pointer)
            self.monologue_history.append(monologue)
            if len(self.monologue_history) > self.max_monologue_history:
                self.monologue_history.pop(0)
        
        # 6. 应用 STDP
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
        构建潜意识续写 Prompt (使用ChatML格式)
        
        独白应该简洁、有意义
        """
        # 构建思维触发
        if self.thought_seed:
            trigger = self.thought_seed
            self.thought_seed = ""  # 使用后清空
        elif self.monologue_history:
            trigger = self.monologue_history[-1][-30:] if len(self.monologue_history[-1]) > 30 else self.monologue_history[-1]
        else:
            trigger = "思考中"
        
        # 简洁的系统消息
        system_msg = "你是一个AI助手，正在进行内部思考。用中文简短表达你的想法。"
        
        # 简洁的用户消息
        user_msg = trigger
        
        # 使用ChatML格式
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        try:
            prompt = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # 回退
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt

    def _generate_with_hidden_state(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2
    ) -> tuple:
        """生成文本并提取隐藏状态"""
        try:
            # 编码输入
            input_ids = self.model.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 定义停止token
            eos_token_id = self.model.tokenizer.eos_token_id
            im_end_token_id = 151645  # <|im_end|>
            stop_token_ids = [eos_token_id, im_end_token_id]
            
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
                    pad_token_id=eos_token_id,
                    eos_token_id=stop_token_ids
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

    def _store_with_real_features(self, monologue: str, hidden_state: Optional[torch.Tensor], is_core: bool = False, semantic_pointer: str = None):
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
            
            # 语义指针（优先使用传入的，否则自动提取）
            if semantic_pointer is None:
                semantic_pointer = monologue[:30] if len(monologue) > 30 else monologue
            
            # 存储到海马体
            current_time = int(time.time() * 1000)
            memory_id = self.hippocampus.encode(
                features=features,
                token_id=hash(monologue) % 100000,
                timestamp=current_time,
                context=[{'content': monologue, 'semantic_pointer': semantic_pointer, 'is_core': is_core}]
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
            
            # 学习率与固化率（增强版）
            lr = 0.02  # 学习率（从0.005增加到0.02）
            consolidation_rate = 0.001  # 固化率（从0.0001增加到0.001）
            max_dynamic_ratio = 0.10  # 10% 的范数限制（从5%增加到10%）
            
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
                
                # 归一化向量
                v_out = v_out / (v_out.norm() + 1e-6)
                v_in = v_in / (v_in.norm() + 1e-6)
                
                delta_w = torch.outer(v_out, v_in) * lr
                
                # 2. 加入随机性探索 (Noise Exploration) - 增强版
                delta_w += torch.randn_like(delta_w) * (lr * 0.3)  # 从0.1增加到0.3
                
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
        
        # 2. 召回海马体记忆
        memory_context = ""
        try:
            input_ids = self.model.tokenizer.encode(user_input[:50], return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model.model.base_model.get_input_embeddings()(input_ids)
            query_features = embeddings.mean(dim=1).squeeze(0)
            if query_features.shape[0] != 1024:
                query_features = self.feature_adapter(query_features.unsqueeze(0)).squeeze(0)
            recalled_memories = self.hippocampus.recall(query_features, topk=2)
            if recalled_memories:
                memory_pointers = [m['semantic_pointer'] for m in recalled_memories if m.get('semantic_pointer')]
                if memory_pointers:
                    memory_context = "相关记忆: " + " | ".join(memory_pointers[:2])
        except:
            pass
        
        # 3. 产生潜意识独白
        monologue = await asyncio.to_thread(self._generate_spontaneous_monologue, 30, 0.9)
        yield {"type": "monologue", "content": monologue}
        
        # 4. 生成正式回复流
        prompt = self._format_chat_prompt(user_input, history, monologue, memory_context)
        
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

    def _format_chat_prompt(self, user_input: str, history: List[Dict[str, str]] = None, monologue: str = "", memory_context: str = "") -> str:
        """
        构建对话 Prompt (类人模式) - 优化版本
        
        优化：
        1. 使用 ChatML 格式，避免标签泄露到输出
        2. 简化 prompt 结构，提高 0.8B 模型理解
        3. 记忆和思维以自然语言方式注入
        """
        # 构建系统消息
        system_parts = ["你是一个有帮助的AI助手。请根据上下文和记忆准确回答用户问题。"]
        
        # 注入海马体召回的记忆（自然语言方式）
        if memory_context:
            system_parts.append(f"【相关记忆】{memory_context}")
        
        # 注入最近的潜意识（独白）
        if monologue and len(monologue) > 3:
            system_parts.append(f"【当前思考】{monologue}")
        
        system_msg = "\n".join(system_parts)
        
        # 使用 ChatML 格式
        messages = [
            {"role": "system", "content": system_msg}
        ]
        
        # 历史记录 (只取最近 2 轮以减轻 0.8B 负担)
        if history:
            for msg in history[-2:]:
                role = "user" if msg['role'] == 'user' else "assistant"
                messages.append({"role": role, "content": msg['content']})
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        # 使用 tokenizer 的 chat template
        try:
            prompt = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # 回退到简单格式
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            if history:
                for msg in history[-2:]:
                    role = "user" if msg['role'] == 'user' else "assistant"
                    prompt += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt



    def _clean_output(self, text: str) -> str:
        """
        清理输出文本，移除可能的标签泄露
        
        优化：
        1. 移除 ChatML 标签
        2. 移除自定义标签
        3. 移除多余的空白行
        """
        # 移除 ChatML 标签
        tags_to_remove = [
            '<|im_start|>', '<|im_end|>',
            '<system>', '</system>',
            '<memory>', '</memory>',
            '<thought>', '</thought>',
            '<user>', '</user>',
            '<assistant>', '</assistant>'
        ]
        
        for tag in tags_to_remove:
            text = text.replace(tag, '')
        
        # 移除角色标记
        text = re.sub(r'^(system|user|assistant):\s*', '', text, flags=re.MULTILINE)
        
        # 移除多余的空白行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 移除开头和结尾的空白
        text = text.strip()
        
        return text
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
            'monologue': {
                'thought_state': self._internal_thought_state.value if hasattr(self, '_internal_thought_state') else 'unknown',
                'emotion_state': self._internal_emotion_state.value if hasattr(self, '_internal_emotion_state') else 'unknown',
                'history_count': len(self.monologue_history),
                'engine_active': self.monologue_engine is not None
            } if self.monologue_engine else {
                'thought_state': 'simplified',
                'emotion_state': 'unknown',
                'history_count': len(self.monologue_history),
                'engine_active': False
            },
            'system': {
                'total_cycles': self.cycle_count, 
                'device': self.device,
                'has_thought_state': self.current_thought_state is not None
            }
        }

    def save_checkpoint(self, path: str):
        """保存检查点 (兼容旧版，仅用于快速测试)"""
        checkpoint = {
            'monologue_history': self.monologue_history,
            'cycle_count': self.cycle_count,
            'total_stdp_updates': self.total_stdp_updates,
            'thought_seed': self.thought_seed
        }
        torch.save(checkpoint, path)
        print(f"[BrainAI] 检查点已保存：{path}")

    def save_state(self, path: str):
        """保存 AI 的完整状态 (睡眠固化)"""
        print(f"[BrainAI] 正在固化记忆与意识状态...")
        try:
            state = {
                'model_state_dict': self.model.model.state_dict(),
                'hippocampus_state': self.hippocampus.get_state(),
                'monologue_history': self.monologue_history,
                'cycle_count': self.cycle_count,
                'total_stdp_updates': self.total_stdp_updates,
                'current_thought_state': self.current_thought_state
            }
            torch.save(state, path)
            print(f"[BrainAI] ✓ 完整状态已保存到: {path}")
        except Exception as e:
            logger.error(f"状态保存失败: {e}")

    def load_state(self, path: str) -> bool:
        """加载 AI 的完整状态 (唤醒)"""
        try:
            import os
            if not os.path.exists(path):
                print(f"[BrainAI] 未找到状态文件 {path}，将执行创世注入。")
                return False
            
            print(f"[BrainAI] 正在从 {path} 唤醒意识...")
            state = torch.load(path, map_location=self.device)
            
            # 1. 恢复模型权重 (包括静态和动态)
            self.model.model.load_state_dict(state['model_state_dict'])
            
            # 2. 恢复海马体
            self.hippocampus.set_state(state['hippocampus_state'])
            
            # 3. 恢复意识状态
            self.monologue_history = state.get('monologue_history', [])
            self.cycle_count = state.get('cycle_count', 0)
            self.total_stdp_updates = state.get('total_stdp_updates', 0)
            self.current_thought_state = state.get('current_thought_state', None)
            
            print(f"[BrainAI] ✓ 意识已成功唤醒。")
            return True
            
        except Exception as e:
            logger.error(f"状态加载失败: {e}。将重新初始化。")
            return False

    def _seed_genesis_memory(self):
        """创世注入：读取 whoami.md 并形成核心记忆"""
        print("[BrainAI] 正在注入创世记忆 (Genesis Seeding)...")
        try:
            with open("whoami.md", "r", encoding="utf-8") as f:
                content = f.read()
            
            # 将 Markdown 内容按标题分割成块
            blocks = content.split("## ")
            for block in blocks[1:]: # 跳过第一个空块
                title, text = block.split("\n", 1)
                title = title.strip()
                text = text.strip()
                
                if not text:
                    continue
                
                # 对每一块核心记忆进行“思考”和“存储”
                prompt = f"关于我的'{title}', 我需要理解：{text[:100]}... 我的想法是："
                output, hidden_state = self._generate_with_hidden_state(prompt, max_tokens=50)
                
                # 存储原文和思考过程
                self._store_with_real_features(f"核心记忆：{title} - {text}", hidden_state, is_core=True)
                self._store_with_real_features(f"(我的思考：{output})", hidden_state, is_core=True)
                
                # 更新思维状态
                if hidden_state is not None:
                    self.current_thought_state = hidden_state
            
            print("[BrainAI] ✓ 创世记忆注入完成。")
            
        except FileNotFoundError:
            print("[BrainAI] ⚠️ 未找到 whoami.md，AI 将以无身份状态启动。")
        except Exception as e:
            logger.error(f"创世记忆注入失败: {e}")

    def _inject_wakeup_memory(self):
        """注入唤醒时间和日期记忆"""
        from datetime import datetime
        now = datetime.now()
        wakeup_time_str = now.strftime("%Y年%m月%d日 %H:%M:%S")
        
        # 形成一个关于时间的思考
        prompt = f"我刚刚“醒来”，现在是 {wakeup_time_str}。"
        output, hidden_state = self._generate_with_hidden_state(prompt, max_tokens=20)
        
        # 存储这个唤醒事件
        self._store_with_real_features(f"唤醒事件：{prompt} {output}", hidden_state)
        
        # 将其设为当前思维种子
        self.thought_seed = f"我刚在 {wakeup_time_str} 醒来，我记得..."
        
        print(f"[BrainAI] ✓ 已注入唤醒记忆：{wakeup_time_str}")


def create_brain_ai(config, device=None):
    return BrainAIInterface(config, device=device)
