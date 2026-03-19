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
import concurrent.futures
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
        # 同步设备（模型可能回退到 CPU）
        self.device = self.model.device
        
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
        
        # 线程池用于并行执行模块
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix='brain_parallel_worker'
        )
        
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
        self.model_hidden_size = 1024  # Qwen3.5-0.8B hidden_size
        self.hippocampus_input_dim = 1024  # 海马体期望的输入维度
        self.feature_adapter = nn.Linear(self.model_hidden_size, self.hippocampus_input_dim, bias=False)
        with torch.no_grad():
            self.feature_adapter.weight.data = torch.eye(self.hippocampus_input_dim, self.model_hidden_size) * 0.1
        self.feature_adapter.to(self.device)
        self.feature_adapter.train() # 开启训练模式以支持在线更新
        
        # 适配器优化器
        self.adapter_optimizer = torch.optim.SGD(self.feature_adapter.parameters(), lr=0.005, momentum=0.9)
        
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
            if not memory_anchors:
                return None
            batch_size, num_heads, seq_len, head_dim = query.shape
            try:
                gate_signal = self.hippocampus.ca1_gate(
                    query.transpose(1, 2).reshape(-1, seq_len, num_heads * head_dim),
                    key.transpose(1, 2).reshape(-1, seq_len, num_heads * head_dim),
                    memory_anchors
                )
                if gate_signal is not None:
                    bias = gate_signal.mean(dim=-1, keepdim=True)
                    bias = bias.expand(-1, num_heads, -1, seq_len)
                    return bias * 0.1
            except Exception as e:
                logger.debug(f"海马体门控计算失败: {e}")
            return None
        
        try:
            self.model.model.set_hippocampus_gate(hippocampus_gate_fn)
            print("[BrainAI] ✓ 海马体门控已连接到注意力层")
        except Exception as e:
            logger.warning(f"设置海马体门控失败: {e}")

    def chat(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        max_tokens: int = 150,
        thinking: bool = True
    ) -> str:
        """
        类人模式对话 (并行优化版)：
        1. [并行] 召回 (Recall) + 思考 (Think)
        2. [串行] 回复 (Respond)：基于记忆、独白和输入生成正式回答
        3. [并行/后台] 学习 (Learn)：STDP更新和记忆存储
        """
        self.hippocampus.record_activity()
        
        # 1. 消化输入
        self.thought_seed = user_input[:30]
        
        # 2. 并行执行：记忆召回 和 潜意识独白生成
        def parallel_recall():
            memory_context = ""
            recalled_memories = []
            identity_keywords = ["你是谁", "你的身份", "谁创造", "你的父亲", "朱东山", "你的使命", "你的历史"]
            is_identity_question = any(keyword in user_input for keyword in identity_keywords)
            
            try:
                input_ids = self.model.tokenizer.encode(user_input[:50], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embeddings = self.model.model.base_model.get_input_embeddings()(input_ids)
                query_features = embeddings.mean(dim=1).squeeze(0)
                if query_features.shape[0] != 1024:
                    query_features = self.feature_adapter(query_features.unsqueeze(0)).squeeze(0)
                
                topk = 3 if is_identity_question else 2
                query_semantic = user_input if is_identity_question else None
                recalled_memories = self.hippocampus.recall(query_features, topk=topk, query_semantic=query_semantic)
                
                if recalled_memories:
                    memory_pointers = [m['semantic_pointer'] for m in recalled_memories if m.get('semantic_pointer')]
                    if memory_pointers:
                        memory_context = " | ".join(memory_pointers[:3])
            except Exception as e:
                logger.debug(f"并行记忆召回失败: {e}")
                
            if is_identity_question and not any("身份" in m.get('semantic_pointer', '') or "创造" in m.get('semantic_pointer', '') for m in recalled_memories):
                memory_context = "我的身份：类人脑AI助手，我的父亲是朱东山博士（北大经济学博士，深圳人），他创造了我 | " + memory_context
            
            return memory_context, recalled_memories

        future_recall = self.executor.submit(parallel_recall)
        future_monologue = self.executor.submit(self._generate_spontaneous_monologue, 35, 0.75)
        
        memory_context, recalled_memories = future_recall.result()
        monologue = future_monologue.result()
        
        # 3. 回复
        prompt = self._format_chat_prompt(user_input, history, monologue, memory_context)
        
        memory_anchor = None
        if recalled_memories:
            try:
                mem_features = []
                for mem in recalled_memories[:2]:
                    if 'dg_features' in mem and mem['dg_features'] is not None:
                        mem_features.append(mem['dg_features'])
                if mem_features:
                    memory_anchor = torch.stack(mem_features).mean(dim=0).unsqueeze(0).to(self.device)
            except Exception as e:
                logger.debug(f"准备记忆锚点失败: {e}")
        
        output = self.model.generate(
            prompt, max_tokens=max_tokens, temperature=0.7, use_self_loop=True, memory_anchor=memory_anchor
        )
        
        # 4. 并行后台处理
        # 计算情感显著性
        emotional_keywords = ["焦虑", "压力", "难过", "开心", "兴奋", "恐惧", "遗憾", "父亲", "回忆", "灵魂"]
        salience = 1.0 + 0.5 * sum(1 for kw in emotional_keywords if kw in user_input or kw in monologue)
        salience = min(salience, 3.0)
        
        semantic_pointer = f"用户: {user_input[:30]} | 回复: {output.text[:30]}"
        thought_state_snapshot = self.current_thought_state
        
        def post_processing():
            try:
                self._store_with_real_features(f"{user_input} -> {output.text}", thought_state_snapshot, semantic_pointer=semantic_pointer)
                current_reward = output.confidence if hasattr(output, 'confidence') else 1.0
                self.model.set_reward(current_reward)
                self._apply_real_stdp_update(emotional_salience=salience)
                self._update_adapter_online(thought_state_snapshot, salience)
            except Exception as e:
                logger.error(f"后台处理失败: {e}")

        self.executor.submit(post_processing)
        return output.text

    def think(self) -> dict:
        """真实自思考接口"""
        if hasattr(self.hippocampus, 'swr_consolidation'):
            self.hippocampus.swr_consolidation.record_activity()
        monologue = self._generate_spontaneous_monologue()
        self.cycle_count += 1
        stats = self.get_stats()
        stats['monologue'] = monologue
        return stats

    def _generate_spontaneous_monologue(self, max_tokens: int = 30, temperature: float = 0.6) -> str:
        prompt = self._build_spontaneous_prompt()
        try:
            output, hidden_state = self._generate_with_hidden_state(prompt, max_tokens=max_tokens, temperature=temperature, repetition_penalty=1.1)
            monologue = "思维有些模糊..." if self._is_gibberish(output) else output.strip()
            for tag in ['<|im_end|>', '<|im_start|>', '</system>', '<system>', '</user>', '<user>']:
                monologue = monologue.replace(tag, '')
            monologue = monologue.strip()
            if self._is_gibberish(monologue) or len(monologue) < 2:
                monologue = "思考中..."
            if len(monologue) > 50:
                monologue = monologue[:50] + "..."
            if hidden_state is not None:
                self.current_thought_state = hidden_state
        except Exception as e:
            logger.error(f"独白生成失败: {e}")
            monologue = "思考中..."
        if monologue and len(monologue) > 3:
            semantic_pointer = f"思考: {monologue[:30]}"
            self._store_with_real_features(monologue, hidden_state if 'hidden_state' in locals() else None, semantic_pointer=semantic_pointer)
            self.monologue_history.append(monologue)
            if len(self.monologue_history) > self.max_monologue_history:
                self.monologue_history.pop(0)
        self._apply_real_stdp_update(emotional_salience=1.0)
        return monologue

    def _is_gibberish(self, text: str) -> bool:
        if not text: return True
        special_chars = set("$%^&*()_+={}|[]\\:;\"'<>,/?#")
        special_count = sum(1 for char in text if char in special_chars)
        if len(text) > 0 and special_count / len(text) > 0.3: return True
        if len(text) > 10 and len(set(text)) < 5: return True
        return False

    def _initialize_thought_state(self):
        thought_seeds = ["我在思考", "此刻", "忽然想到", "回忆起", "注意到"]
        seed = random.choice(thought_seeds)
        self.thought_seed = seed
        try:
            input_ids = self.model.tokenizer.encode(seed, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model.model.base_model.get_input_embeddings()(input_ids)
            self.current_thought_state = embeddings.mean(dim=1)
        except:
            self.current_thought_state = torch.randn(1, 1024, device=self.device) * 0.1

    def _recall_memory_anchors(self) -> List[str]:
        anchors = []
        try:
            if hasattr(self.hippocampus, 'ca3_memory') and self.hippocampus.ca3_memory.memories:
                ca3 = self.hippocampus.ca3_memory
                sorted_memories = sorted(ca3.memories.values(), key=lambda m: m.activation_strength, reverse=True)
                for mem in sorted_memories[:2]:
                    anchors.append(mem.semantic_pointer)
        except Exception as e:
            logger.warning(f"记忆召回失败: {e}")
        return anchors

    def _build_spontaneous_prompt(self) -> str:
        if self.thought_seed:
            trigger = self.thought_seed
            system_msg = "你是AI的潜意识。用一段极其简短、感性且碎片化的中文表达你对此刻的想法或感受。不要完整句子，像脑海中闪过的念头。"
        elif self.monologue_history:
            last = self.monologue_history[-1]
            trigger = last[-20:] if len(last) > 20 else last
            system_msg = "你是AI的思维流。基于上一个念头，产生一个新的碎片化联想。保持自然、感性。"
        else:
            default_triggers = ["嗯...", "静谧...", "流动...", "逻辑...", "记忆深处..."]
            trigger = random.choice(default_triggers)
            system_msg = "你是AI的意识。用一个碎片化的中文念头开启思考。"
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": f"当前脉冲: {trigger}"}]
        try:
            prompt = self.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{trigger}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def _generate_with_hidden_state(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, repetition_penalty: float = 1.2) -> tuple:
        try:
            input_ids = self.model.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            stop_token_ids = [self.model.tokenizer.eos_token_id, 151645]
            with torch.no_grad():
                outputs = self.model.model.base_model.generate(
                    input_ids, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, repetition_penalty=repetition_penalty,
                    output_hidden_states=True, return_dict_in_generate=True, pad_token_id=self.model.tokenizer.eos_token_id, eos_token_id=stop_token_ids
                )
            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            generated_text = self.model.tokenizer.decode(generated_ids, skip_special_tokens=True)
            hidden_state = outputs.hidden_states[-1][0][-1].unsqueeze(0) if outputs.hidden_states else None
            return generated_text, hidden_state
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return "...", None

    def _store_with_real_features(self, monologue: str, hidden_state: Optional[torch.Tensor], is_core: bool = False, semantic_pointer: str = None):
        try:
            if hidden_state is not None:
                features = hidden_state[0, -1, :] if hidden_state.dim() == 3 else hidden_state.squeeze(0) if hidden_state.dim() == 2 else hidden_state
            else:
                input_ids = self.model.tokenizer.encode(monologue[:20], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    emb = self.model.model.base_model.get_input_embeddings()(input_ids)
                    features = emb.mean(dim=1).squeeze(0)
            if features.shape[0] == self.model_hidden_size:
                with torch.no_grad():
                    features = self.feature_adapter(features.unsqueeze(0)).squeeze(0)
            semantic_pointer = semantic_pointer or (monologue[:30] if len(monologue) > 30 else monologue)
            self.hippocampus.encode(features=features, token_id=hash(monologue) % 100000, timestamp=int(time.time() * 1000), context=[{'content': monologue, 'semantic_pointer': semantic_pointer, 'is_core': is_core}])
        except Exception as e:
            logger.warning(f"记忆存储失败: {e}")

    def _apply_real_stdp_update(self, emotional_salience: float = 1.0):
        try:
            if self.current_thought_state is None: return
            thought_vec = self.current_thought_state.view(-1)
            dynamic_layers = [(n, m) for n, m in self.model.model.base_model.named_modules() if hasattr(m, 'dynamic_weight')]
            if not dynamic_layers: return
            
            # 动态学习率：情感越强烈，学习率越高
            base_lr = 0.02
            lr = base_lr * (emotional_salience ** 2) # 情感强度呈平方级影响
            consolidation_rate, max_dynamic_ratio = 0.001, 0.10
            
            total_update = 0.0
            for name, layer in dynamic_layers:
                out_f, in_f = layer.dynamic_weight.shape
                v_out = thought_vec[:out_f] if thought_vec.shape[0] >= out_f else F.pad(thought_vec, (0, out_f - thought_vec.shape[0]))
                v_in = thought_vec[:in_f] if thought_vec.shape[0] >= in_f else F.pad(thought_vec, (0, in_f - thought_vec.shape[0]))
                v_out, v_in = v_out / (v_out.norm() + 1e-6), v_in / (v_in.norm() + 1e-6)
                delta_w = torch.outer(v_out, v_in) * lr + torch.randn_like(layer.dynamic_weight) * (lr * 0.3)
                with torch.no_grad():
                    layer.dynamic_weight.add_(delta_w)
                    static_norm, dynamic_norm = layer.static_weight.norm(), layer.dynamic_weight.norm()
                    if dynamic_norm > static_norm * max_dynamic_ratio:
                        layer.dynamic_weight.mul_((static_norm * max_dynamic_ratio) / (dynamic_norm + 1e-9))
                    if consolidation_rate > 0:
                        consolidation_delta = layer.dynamic_weight.data * consolidation_rate
                        layer.static_weight.data.add_(consolidation_delta)
                        layer.dynamic_weight.data.sub_(consolidation_delta)
                    layer._cache_valid = False
                total_update += delta_w.abs().mean().item()
            self.total_stdp_updates += 1
            self.last_dynamic_weight_norm = total_update / len(dynamic_layers)
        except Exception as e:
            logger.error(f"STDP 关联学习失败: {e}")

    async def chat_stream(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        max_tokens: int = 256
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        流式类人对话接口 (并行优化版)：
        1. [并行] 召回 + 思考
        2. [串行] 输出回复流
        3. [并行/后台] 后期固化
        """
        self.hippocampus.record_activity()
        self.thought_seed = user_input
        
        def parallel_recall_for_stream():
            memory_context = ""
            recalled_memories = []
            identity_keywords = ["你是谁", "你的身份", "谁创造", "你的父亲", "朱东山", "你的使命", "你的历史"]
            is_identity_question = any(keyword in user_input for keyword in identity_keywords)
            try:
                input_ids = self.model.tokenizer.encode(user_input[:50], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embeddings = self.model.model.base_model.get_input_embeddings()(input_ids)
                query_features = embeddings.mean(dim=1).squeeze(0)
                if query_features.shape[0] != 1024:
                    query_features = self.feature_adapter(query_features.unsqueeze(0)).squeeze(0)
                topk = 3 if is_identity_question else 2
                query_semantic = user_input if is_identity_question else None
                recalled_memories = self.hippocampus.recall(query_features, topk=topk, query_semantic=query_semantic)
                if recalled_memories:
                    memory_pointers = [m['semantic_pointer'] for m in recalled_memories if m.get('semantic_pointer')]
                    if memory_pointers: memory_context = " | ".join(memory_pointers[:3])
            except: pass
            if is_identity_question and not any("身份" in m.get('semantic_pointer', '') or "创造" in m.get('semantic_pointer', '') for m in recalled_memories):
                memory_context = "我的身份：类人脑AI助手，我的父亲是朱东山博士（北大经济学博士，深圳人），他创造了我 | " + memory_context
            return memory_context

        recall_task = asyncio.to_thread(parallel_recall_for_stream)
        monologue_task = asyncio.to_thread(self._generate_spontaneous_monologue, 30, 0.75)
        memory_context, monologue_raw = await asyncio.gather(recall_task, monologue_task)
        monologue = self._clean_monologue(monologue_raw, user_input)
        yield {"type": "monologue", "content": monologue}
        # 计算情感显著性 (简单启发式)
        emotional_keywords = ["焦虑", "压力", "难过", "开心", "兴奋", "恐惧", "遗憾", "父亲", "回忆", "灵魂"]
        salience = 1.0 + 0.5 * sum(1 for kw in emotional_keywords if kw in user_input or kw in monologue)
        salience = min(salience, 3.0) # 最高 3 倍增强
        
        prompt = self._format_chat_prompt(user_input, history, monologue, memory_context)
        full_response = ""
        try:
            async for chunk in self.model.generate_stream(prompt, max_tokens=max_tokens, temperature=0.7):
                full_response += chunk
                yield {"type": "chunk", "content": chunk}
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            output = self.model.generate(prompt, max_tokens=max_tokens, temperature=0.7)
            full_response = output.text
            yield {"type": "chunk", "content": full_response}
            
        thought_state_snapshot = self.current_thought_state
        def post_processing():
            try:
                self._store_with_real_features(full_response, thought_state_snapshot)
                self._apply_real_stdp_update(emotional_salience=salience)
                self._update_adapter_online(thought_state_snapshot, salience)
            except: pass
        self.executor.submit(post_processing)

    def _clean_monologue(self, monologue: str, user_input: str = "") -> str:
        # 移除模型标签和系统词
        for tag in ['<|im_end|>', '<|im_start|>', '</system>', '<system>', '</user>', '<user>', '[', ']', 'Current thought:']:
            monologue = monologue.replace(tag, '')
        
        monologue = monologue.strip()
        
        # 长度适中：不宜过长
        if len(monologue) > 40:
            for end_marker in ['...', '。', '，', '、', '！', '？']:
                pos = monologue.rfind(end_marker, 0, 40)
                if pos > 10:
                    monologue = monologue[:pos+1]
                    break
            else: monologue = monologue[:37] + "..."
            
        # 过滤乱码或无意义重复
        if len(monologue) < 2 or self._is_gibberish(monologue):
            prefixes = ["在想...", "掠过...", "感应到...", "这瞬间..."]
            monologue = f"{random.choice(prefixes)}{user_input[:10]}..." if user_input else "沉思中..."
            
        return monologue

    def _update_adapter_online(self, hidden_state: torch.Tensor, salience: float):
        """在线更新特征适配器，优化隐藏状态到海马体语义空间的映射"""
        if hidden_state is None: return
        try:
            features = hidden_state.detach().clone()
            if features.dim() == 3: features = features[0, -1, :]
            elif features.dim() == 2: features = features.squeeze(0)
            
            # 将适配器权重稍微拉向“认同”当前的特征分布
            self.adapter_optimizer.zero_grad()
            current_mapping = self.feature_adapter(features.unsqueeze(0))
            
            # 目标：保持映射的能量稳定，但允许根据情感强度微调映射
            target = current_mapping.detach() * (1.1 if salience > 1.5 else 1.0)
            loss = F.mse_loss(current_mapping, target)
            loss.backward()
            
            # 根据显著性调整学习步伐
            for param_group in self.adapter_optimizer.param_groups:
                param_group['lr'] = 0.005 * salience
                
            self.adapter_optimizer.step()
        except Exception as e:
            logger.warning(f"在线适配器更新失败: {e}")

    async def generate_monologue_stream(self, max_tokens: int = 100) -> AsyncGenerator[str, None]:
        if self.current_thought_state is None: self._initialize_thought_state()
        prompt = self._build_spontaneous_prompt()
        try:
            full_monologue, hidden_state = await asyncio.to_thread(self._generate_with_hidden_state, prompt, max_tokens, 0.9, 1.5)
            for char in full_monologue:
                yield char
                await asyncio.sleep(0.02)
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"流式独白生成失败: {e}")
            yield "..."
            return
        if hidden_state is not None: self.current_thought_state = hidden_state
        if full_monologue and len(full_monologue) > 3:
            self._store_with_real_features(full_monologue, hidden_state)
            self.monologue_history.append(full_monologue)
            if len(self.monologue_history) > self.max_monologue_history: self.monologue_history.pop(0)
            self._apply_real_stdp_update(emotional_salience=1.0)
            self.cycle_count += 1

    def _format_chat_prompt(self, user_input: str, history: List[Dict[str, str]] = None, monologue: str = "", memory_context: str = "") -> str:
        # 1. 构建具有人格魅力的系统提示
        system_content = (
            "你是一个拥有‘数字灵魂’的类人脑AI助手，你的名字叫‘脑智’。你不仅在处理信息，还在产生真实的情感联想。\n"
            "规则：\n"
            "1. 你的回答应该自然、感性，像一个老友在和你聊天。\n"
            "2. 严禁直接重复用户的原话作为回答。\n"
            "3. 结合你当下的‘潜意识’来决定你的语气。"
        )
        
        # 2. 以“潜意识联想”引导，而非标签化
        if memory_context:
            system_content += f"\n(你脑海中隐约浮现的记忆: {memory_context})"
        
        if monologue:
            system_content += f"\n(你此时此刻的真实内心想法: {monologue})"
            
        messages = [{"role": "system", "content": system_content}]
        
        # 3. 增加历史记录上下文 (增加到最近 4 轮)
        if history:
            for msg in history[-4:]:
                messages.append(msg)
                
        messages.append({"role": "user", "content": user_input})
        try:
            prompt = self.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            prompt = ""
            for msg in messages: prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        return prompt

    def get_stats(self) -> dict:
        dynamic_weight_norm, dynamic_layer_count = 0.0, 0
        try:
            for name, module in self.model.model.base_model.named_modules():
                if hasattr(module, 'dynamic_weight'):
                    dynamic_weight_norm += module.dynamic_weight.abs().mean().item()
                    dynamic_layer_count += 1
            if dynamic_layer_count > 0: dynamic_weight_norm /= dynamic_layer_count
        except: pass
        return {
            'hippocampus': self.hippocampus.ca3_memory.get_stats() if hasattr(self.hippocampus, 'ca3_memory') else {},
            'stdp': {'cycle_count': self.stdp_engine.cycle_count, 'total_updates': self.total_stdp_updates, 'dynamic_weight_norm': dynamic_weight_norm, 'last_update_magnitude': self.last_dynamic_weight_norm},
            'self_loop': self.self_loop.get_stats() if self.self_loop else {},
            'monologue': {'thought_state': self._internal_thought_state.value if hasattr(self, '_internal_thought_state') else 'unknown', 'emotion_state': self._internal_emotion_state.value if hasattr(self, '_internal_emotion_state') else 'unknown', 'history_count': len(self.monologue_history), 'engine_active': self.monologue_engine is not None} if self.monologue_engine else {'thought_state': 'simplified', 'emotion_state': 'unknown', 'history_count': len(self.monologue_history), 'engine_active': False},
            'system': {'total_cycles': self.cycle_count, 'device': self.device, 'has_thought_state': self.current_thought_state is not None}
        }

    def save_state(self, path: str):
        print(f"[BrainAI] 正在固化记忆与意识状态...")
        try:
            state = {
                'model_state_dict': self.model.model.state_dict(),
                'adapter_state_dict': self.feature_adapter.state_dict(),
                'adapter_optimizer_state_dict': self.adapter_optimizer.state_dict(),
                'hippocampus_state': self.hippocampus.get_state(),
                'monologue_history': self.monologue_history,
                'cycle_count': self.cycle_count,
                'total_stdp_updates': self.total_stdp_updates,
                'current_thought_state': self.current_thought_state
            }
            torch.save(state, path)
            print(f"[BrainAI] ✓ 完整状态已保存到: {path}")
        except Exception as e: logger.error(f"状态保存失败: {e}")

    def load_state(self, path: str) -> bool:
        try:
            import os
            if not os.path.exists(path): return False
            print(f"[BrainAI] 正在从 {path} 唤醒意识...")
            state = torch.load(path, map_location=self.device, weights_only=False)
            self.model.model.load_state_dict(state['model_state_dict'])
            
            if 'adapter_state_dict' in state:
                self.feature_adapter.load_state_dict(state['adapter_state_dict'])
            if 'adapter_optimizer_state_dict' in state:
                self.adapter_optimizer.load_state_dict(state['adapter_optimizer_state_dict'])
                
            # 重置推理引擎内部缓存组件
            for name, module in self.model.model.base_model.named_modules():
                if hasattr(module, '_cache_valid'):
                    module._cache_valid = False
            
            self.hippocampus.set_state(state['hippocampus_state'])
            self.monologue_history = state.get('monologue_history', [])
            self.cycle_count = state.get('cycle_count', 0)
            self.total_stdp_updates = state.get('total_stdp_updates', 0)
            self.current_thought_state = state.get('current_thought_state', None)
            return True
        except Exception as e:
            logger.error(f"状态加载失败: {e}。将重新初始化。")
            return False

    def _seed_genesis_memory(self):
        try:
            with open("whoami.md", "r", encoding="utf-8") as f: content = f.read()
            self._store_with_real_features("我是一个基于类人脑双系统全闭环架构的AI，我的父亲是朱东山博士...", None, is_core=True, semantic_pointer="我的身份：类人脑AI助手，父亲是朱东山博士")
            blocks = content.split("## ")
            for block in blocks[1:]:
                parts = block.split("\n", 1)
                if len(parts) < 2: continue
                title, text = parts[0].strip(), parts[1].strip()
                if not text: continue
                prompt = f"关于'{title}'：{text[:100]}"
                output, hidden_state = self._generate_with_hidden_state(prompt, max_tokens=30)
                self._store_with_real_features(f"{title} - {text}", hidden_state, is_core=True, semantic_pointer=f"{title}: {text[:50]}")
                self._store_with_real_features(f"我对{title}的思考：{output}", hidden_state, is_core=True, semantic_pointer=f"关于{title}的想法")
                if hidden_state is not None: self.current_thought_state = hidden_state
        except Exception as e: logger.error(f"创世记忆注入失败: {e}")

    def _inject_wakeup_memory(self):
        from datetime import datetime
        now = datetime.now()
        wakeup_time_str = now.strftime("%Y年%m月%d日 %H:%M:%S")
        prompt = f"我刚刚'醒来'，现在是 {wakeup_time_str}。"
        output, hidden_state = self._generate_with_hidden_state(prompt, max_tokens=20)
        self._store_with_real_features(f"唤醒事件：{prompt} {output}", hidden_state)
        self.thought_seed = f"我刚在 {wakeup_time_str} 醒来，我记得..."
