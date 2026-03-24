"""
类人脑双系统全闭环 AI架构 - 生产级核心接口

集成真实的 Qwen3.5-0.8B 模型、海马体系统、STDP 引擎和自闭环优化器。
实现真实的内心独白流（由模型隐藏状态驱动，自发推进）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
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
from core.inner_thought_engine import InnerThoughtEngine, MindState, ThinkingMode
from core.goal_system import GoalSystem, create_goal_system
from core.global_workspace import GlobalWorkspace, create_global_workspace

logger = logging.getLogger(__name__)


@dataclass
class BrainAIOutput:
    """AI输出结果数据结构"""
    text: str
    tokens: List[str]
    confidence: float
    memory_anchors: List[Dict] = field(default_factory=list)
    stdp_stats: Dict = field(default_factory=dict)
    cycle_stats: Dict = field(default_factory=dict)


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
        self.inner_thought_engine = None  # 统一的内心思维独白引擎
        
        # 特征维度适配器（模型 hidden_size -> 海马体输入维度）
        # 必须在使用 model_hidden_size 之前定义
        self.model_hidden_size = 1024  # Qwen3.5-0.8B hidden_size
        self.hippocampus_input_dim = 1024  # 海马体期望的输入维度
        self.feature_adapter = nn.Linear(self.model_hidden_size, self.hippocampus_input_dim, bias=False)
        with torch.no_grad():
            self.feature_adapter.weight.data = torch.eye(self.hippocampus_input_dim, self.model_hidden_size) * 0.1
        self.feature_adapter.to(self.device)
        self.feature_adapter.train() # 开启训练模式以支持在线更新
        
        # 适配器优化器
        self.adapter_optimizer = torch.optim.SGD(self.feature_adapter.parameters(), lr=0.005, momentum=0.9)
        
        # 6. 加载目标系统 (新增)
        from core.goal_system import GoalSystem
        self.goal_system = GoalSystem(hidden_size=self.model_hidden_size, device=self.device)
        print("[BrainAI] [OK] 目标驱动系统已初始化")
        
        # 7. 加载全局工作空间 (新增)
        from core.global_workspace import GlobalWorkspace
        self.global_workspace = GlobalWorkspace(hidden_size=self.model_hidden_size, device=self.device)
        # 设置模型引用，用于获取真实embedding
        self.global_workspace.set_model(self.model)
        print("[BrainAI] [OK] 全局工作空间已初始化")
        
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
        self._internal_thought_state = MindState.RESTING
        self._internal_emotion_state = ThinkingMode.ANALYTICAL
        
        # STDP 学习追踪
        self.total_stdp_updates = 0
        self.last_dynamic_weight_norm = 0.0
        
        # 状态文件路径
        self.state_path = "brain_state.pt"
        
        # ========== 优化：避免重复加载模型权重 ==========
        # 检查是否存在已保存的状态文件
        import os
        if os.path.exists(self.state_path):
            # 如果存在，只加载动态权重部分，不重复加载基础模型
            print(f"[BrainAI] 检测到已保存的意识状态: {self.state_path}")
            print(f"[BrainAI] 跳过重复加载基础模型，仅恢复学习成果...")
            self._load_dynamic_weights_only(self.state_path)
        else:
            # 如果不存在，执行创世注入
            self._seed_genesis_memory()
        
        # 注入唤醒记忆
        self._inject_wakeup_memory()
        
        # 初始化统一的内心思维独白引擎
        try:
            self.inner_thought_engine = InnerThoughtEngine(
                model_interface=self.model,
                hippocampus_system=self.hippocampus,
                self_loop_optimizer=self.self_loop,
                config=config,
                device=self.device
            )
            print("[BrainAI] [OK] 内心思维独白引擎已初始化")
        except Exception as e:
            logger.error(f"内心思维独白引擎初始化失败: {e}")
            raise RuntimeError(f"内心思维独白引擎初始化失败，无法继续: {e}")
        
        # 启动海马体 SWR 监控
        try:
            self.hippocampus.start_swr_monitoring()
            print("[BrainAI] [OK] 海马体 SWR 监控已启动")
        except Exception as e:
            logger.warning(f"SWR 监控启动失败: {e}")
        
        # 初始化目标系统
        try:
            self.goal_system = create_goal_system(hidden_size=self.model_hidden_size, device=self.device)
            print("[BrainAI] [OK] 目标系统已初始化")
        except Exception as e:
            logger.warning(f"目标系统初始化失败: {e}")
            self.goal_system = None
        
        # 初始化全局工作空间
        try:
            self.global_workspace = create_global_workspace(hidden_size=self.model_hidden_size, device=self.device)
            # 设置模型引用，用于获取真实embedding
            self.global_workspace.set_model(self.model)
            print("[BrainAI] [OK] 全局工作空间已初始化")
        except Exception as e:
            logger.warning(f"全局工作空间初始化失败: {e}")
            self.global_workspace = None
        
        # 设置海马体门控函数（连接CA1到注意力层）
        self._setup_hippocampus_gate()
        
        print("[BrainAI] [OK] 高级实现模块集成完成\n")
    
    # ==================== 兼容性属性 ====================
    
    @property
    def monologue_engine(self):
        """兼容性属性: 返回统一的内心思维独白引擎"""
        return self.inner_thought_engine
    
    @property
    def thought_flow_engine(self):
        """兼容性属性: 返回统一的内心思维独白引擎"""
        return self.inner_thought_engine
    
    def _setup_hippocampus_gate(self):
        """设置海马体门控，让CA1门控信号影响注意力"""
        def hippocampus_gate_fn(query, key, memory_anchors):
            # 使用保存的完整记忆字典，而不是传入的tensor
            recalled_memories = getattr(self, '_current_recalled_memories', None)
            if not recalled_memories:
                return None
            batch_size, num_heads, seq_len, head_dim = query.shape
            try:
                gate_signal = self.hippocampus.ca1_gate(
                    query.transpose(1, 2).reshape(-1, seq_len, num_heads * head_dim),
                    key.transpose(1, 2).reshape(-1, seq_len, num_heads * head_dim),
                    recalled_memories  # 传递完整的记忆字典列表
                )
                if gate_signal is not None:
                    bias = gate_signal.mean(dim=-1, keepdim=True)
                    bias = bias.expand(-1, num_heads, -1, seq_len)
                    return bias * 0.5  # 增强门控信号：0.1 → 0.5
            except Exception as e:
                logger.debug(f"海马体门控计算失败: {e}")
            return None
        
        try:
            self.model.model.set_hippocampus_gate(hippocampus_gate_fn)
            print("[BrainAI] [OK] 海马体门控已连接到注意力层")
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
        
        # 1.5 目标推断（新增）
        if hasattr(self, 'goal_system') and self.goal_system:
            try:
                # 使用当前思维状态推断目标
                goal = self.goal_system.infer_goal(user_input, self.current_thought_state)
                logger.debug(f"目标推断: {goal.goal_type.value} - {goal.description}")
            except Exception as e:
                logger.warning(f"目标推断失败: {e}")
        
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
        
        # 存储完整记忆字典供注意力层使用
        self._current_recalled_memories = recalled_memories
        
        # 3. 回复
        # 获取目标信息
        goal_context = ""
        if hasattr(self, 'goal_system') and self.goal_system and self.goal_system.current_goal:
            goal = self.goal_system.current_goal
            goal_context = f"[当前目标：{goal.goal_type.value} - {goal.description}]"
        
        prompt = self._format_chat_prompt(user_input, history, monologue, memory_context, goal_context)
        
        # 准备记忆锚点（保持原有逻辑兼容性）
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
        
        # 3.5 自闭环优化 - 高复杂度任务进行二次优化
        mode = self.self_loop.decide_mode(user_input)
        output_confidence = output.confidence if hasattr(output, 'confidence') else 0.7
        
        if mode in ["self_game", "self_eval"] or output_confidence < 0.6:
            # 高难度任务或低置信度输出，使用自闭环优化
            try:
                context_list = [memory_context] if memory_context else None
                optimized_result = self.self_loop.run(user_input, context=context_list)
                
                # 如果优化后的置信度更高，使用优化结果
                if optimized_result.confidence > output_confidence:
                    logger.debug(f"自闭环优化: {mode} 置信度 {output_confidence:.2f} -> {optimized_result.confidence:.2f}")
                    output.text = optimized_result.output_text
                    output.confidence = optimized_result.confidence
            except Exception as e:
                logger.warning(f"自闭环优化失败，使用原始输出: {e}")
        
        # 4. 并行后台处理
        # 计算情感显著性和核心记忆检测
        emotional_keywords = ["焦虑", "压力", "难过", "开心", "兴奋", "恐惧", "遗憾", "父亲", "回忆", "灵魂"]
        salience = 1.0 + 0.5 * sum(1 for kw in emotional_keywords if kw in user_input or kw in monologue)
        salience = min(salience, 3.0)
        
        # 检测是否为核心记忆（个人身份信息）
        identity_patterns = ["我叫", "我是", "我的名字", "我今年", "我的职业", "我喜欢", "我住"]
        is_core_memory = any(pattern in user_input for pattern in identity_patterns)
        
        # 提取实体信息增强semantic_pointer
        enhanced_pointer = f"用户: {user_input[:30]} | 回复: {output.text[:30]}"
        if is_core_memory:
            # 提取关键实体
            import re
            name_match = re.search(r"我叫(.{2,6})|我的名字(是|叫)(.{2,6})", user_input)
            age_match = re.search(r"我今年(\d+)", user_input)
            job_match = re.search(r"我是(.{2,10})(工程师|医生|老师|学生|设计师|程序员)", user_input)
            hobby_match = re.search(r"我喜欢(.{2,20})", user_input)
            
            entities = []
            if name_match:
                entities.append(f"名字:{name_match.group(1) or name_match.group(3)}")
            if age_match:
                entities.append(f"年龄:{age_match.group(1)}岁")
            if job_match:
                entities.append(f"职业:{job_match.group(0)}")
            if hobby_match:
                entities.append(f"爱好:{hobby_match.group(1)}")
            
            if entities:
                enhanced_pointer = " | ".join(entities) + " | " + enhanced_pointer
        
        thought_state_snapshot = self.current_thought_state
        
        def post_processing():
            try:
                self._store_with_real_features(
                    f"{user_input} -> {output.text}", 
                    thought_state_snapshot, 
                    is_core=is_core_memory,
                    semantic_pointer=enhanced_pointer
                )
                current_reward = output.confidence if hasattr(output, 'confidence') else 1.0
                self.model.set_reward(current_reward)
                self._apply_real_stdp_update(emotional_salience=salience)
                self._update_adapter_online(thought_state_snapshot, salience)
                
                # 更新目标进度（新增）
                if hasattr(self, 'goal_system') and self.goal_system and self.goal_system.current_goal:
                    # 根据目标类型和回复内容判断进度
                    goal = self.goal_system.current_goal
                    if goal.goal_type.value == "remember":
                        # 记忆目标：如果用户提到个人信息，视为完成
                        if is_core_memory and "好的" in output.text or "记住" in output.text:
                            self.goal_system.update_progress(1.0)
                        else:
                            self.goal_system.update_progress(0.5)  # 部分完成
                    elif goal.goal_type.value == "recall":
                        # 回忆目标：如果回复中包含记忆信息
                        if memory_context and len(output.text) > 20:
                            self.goal_system.update_progress(1.0)
                        else:
                            self.goal_system.update_progress(0.3)
                    else:
                        # 其他目标：基础进度
                        self.goal_system.update_progress(0.8)
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

    def _generate_spontaneous_monologue(self, max_tokens: int = 30, temperature: float = 0.55) -> str:
        """
        生成自发独白 - 优化版本
        
        改进：
        - 降低temperature提升稳定性 (0.6 -> 0.55)
        - 增强推理链引导
        """
        prompt = self._build_spontaneous_prompt()
        try:
            # 降低temperature和重复惩罚
            output, hidden_state = self._generate_with_hidden_state(
                prompt, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                repetition_penalty=1.15  # 从1.1提升到1.15，减少重复
            )
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
        """构建理性推理导向的独白提示"""
        # 推理模式选择
        reasoning_modes = [
            ("分析", "从逻辑角度分析"),
            ("推理", "基于已知进行推导"),
            ("质疑", "审视假设和前提"),
            ("综合", "整合多方面信息"),
            ("归纳", "从具体到一般")
        ]
        mode_name, mode_desc = random.choice(reasoning_modes)
        
        if self.thought_seed:
            trigger = self.thought_seed
            system_msg = f"你是一个理性思维的AI。当前任务：{mode_desc}。用简洁的中文表达你的思考过程。保持逻辑性，不要情绪化表达。"
        elif self.monologue_history:
            last = self.monologue_history[-1]
            trigger = last[-30:] if len(last) > 30 else last
            system_msg = f"你是一个理性思维的AI。基于上一个思考，继续{mode_desc}。保持逻辑链条的连贯性。"
        else:
            default_triggers = ["思考起点...", "分析角度...", "推理路径...", "逻辑框架...", "问题本质..."]
            trigger = random.choice(default_triggers)
            system_msg = "你是一个理性思维的AI。开始一个结构化的思考过程。明确你的分析目标。"
        
        messages = [
            {"role": "system", "content": system_msg}, 
            {"role": "user", "content": f"思考对象: {trigger}"}
        ]
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
        """在线更新特征适配器 - 优化版本"""
        if hidden_state is None: return
        try:
            features = hidden_state.detach().clone()
            if features.dim() == 3: features = features[0, -1, :]
            elif features.dim() == 2: features = features.squeeze(0)
            
            # 降低学习率，避免震荡 - 从0.005降到0.002
            self.adapter_optimizer.zero_grad()
            current_mapping = self.feature_adapter(features.unsqueeze(0))
            
            # 降低情感强度影响：从1.1/1.0降到1.05/1.0
            target = current_mapping.detach() * (1.05 if salience > 1.5 else 1.0)
            loss = F.mse_loss(current_mapping, target)
            loss.backward()
            
            # 根据显著性调整学习步伐 - 降低学习率
            for param_group in self.adapter_optimizer.param_groups:
                param_group['lr'] = 0.002 * salience  # 从0.005降到0.002
                
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

    def _format_chat_prompt(self, user_input: str, history: List[Dict[str, str]] = None, monologue: str = "", memory_context: str = "", goal_context: str = "") -> str:
        """格式化对话提示 - 强化推理能力"""
        # 理性推理导向的系统提示 - 增强版
        system_content = (
            "你是'脑智'，一个具备深度推理能力的类人脑AI助手。\n"
            "核心原则：\n"
            "1. 回答要有逻辑结构：分析问题 → 推理过程 → 得出结论\n"
            "2. 使用清晰的思维链，逐步展开你的推理\n"
            "3. 遇到复杂问题时，先分解再逐一分析\n"
            "4. 保持理性客观，避免情绪化表达\n"
            "5. 如果不确定，诚实说明而非编造\n"
            "6. **重要**：根据当前目标调整回复策略\n"
            "7. **关键优化**：优先调用记忆系统中的信息，减少幻觉\n"
            "回答格式建议：\n"
            "- 首先理解问题的核心\n"
            "- 然后展示你的思考过程\n"
            "- 最后给出明确的结论或建议"
        )
        
        # 2. 添加目标上下文（新增）
        if goal_context:
            system_content += f"\n\n{goal_context}\n**请根据目标调整回复策略：**"
            # 根据目标类型添加具体指导
            if "remember" in goal_context.lower():
                system_content += "\n- 用户希望记住信息，请确认并复述关键信息\n- 简洁回复，如：'好的，我记住了您叫张三，来自北京'"
            elif "recall" in goal_context.lower():
                system_content += "\n- 用户在询问记忆，请回忆之前的对话内容\n- 如果记得，直接回答；如果不记得，诚实说明"
        
        # 3. 添加记忆上下文
        if memory_context:
            system_content += f"\n\n[重要记忆 - 请务必参考]\n{memory_context}\n[请根据记忆内容回答用户问题]"
        
        # 4. 添加当前思考状态
        if monologue:
            system_content += f"\n\n[当前思考] {monologue}"
            
        messages = [{"role": "system", "content": system_content}]
        
        # 4. 添加历史对话 (最近4轮)
        if history:
            for msg in history[-4:]:
                messages.append(msg)
                
        messages.append({"role": "user", "content": user_input})
        
        try:
            prompt = self.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            prompt = ""
            for msg in messages:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
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
        
        # 所有核心模块应该都已初始化
        if not self.inner_thought_engine:
            raise RuntimeError("内心思维独白引擎未初始化")
        
        return {
            'hippocampus': self.hippocampus.ca3_memory.get_stats() if hasattr(self.hippocampus, 'ca3_memory') else {},
            'stdp': {'cycle_count': self.stdp_engine.cycle_count, 'total_updates': self.total_stdp_updates, 'dynamic_weight_norm': dynamic_weight_norm, 'last_update_magnitude': self.last_dynamic_weight_norm},
            'self_loop': self.self_loop.get_stats() if self.self_loop else {},
            'monologue': {
                'thought_state': self._internal_thought_state.value if hasattr(self, '_internal_thought_state') else 'unknown',
                'emotion_state': self._internal_emotion_state.value if hasattr(self, '_internal_emotion_state') else 'unknown',
                'history_count': len(self.monologue_history),
                'engine_active': True
            },
            'system': {'total_cycles': self.cycle_count, 'device': self.device, 'has_thought_state': self.current_thought_state is not None}
        }

    def save_state(self, path: str):
        """保存状态 - 优化版本：只保存动态权重，不重复保存基础模型"""
        print(f"[BrainAI] 正在固化记忆与学习成果...")
        try:
            # 只保存动态权重部分
            dynamic_weights = {}
            for name, param in self.model.model.named_parameters():
                if 'dynamic_weight' in name:
                    dynamic_weights[name] = param.data.clone()
            
            state = {
                'dynamic_weights': dynamic_weights,  # 只保存动态权重
                'adapter_state_dict': self.feature_adapter.state_dict(),
                'adapter_optimizer_state_dict': self.adapter_optimizer.state_dict(),
                'hippocampus_state': self.hippocampus.get_state(),
                'monologue_history': self.monologue_history,
                'cycle_count': self.cycle_count,
                'total_stdp_updates': self.total_stdp_updates,
                'current_thought_state': self.current_thought_state
            }
            torch.save(state, path)
            
            # 计算保存的大小
            import os
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[BrainAI] [OK] 学习成果已保存到: {path}")
            print(f"[BrainAI] [INFO] 文件大小: {file_size_mb:.2f} MB (仅动态权重)")
        except Exception as e:
            logger.error(f"状态保存失败: {e}")

    def _load_dynamic_weights_only(self, path: str) -> bool:
        """只加载动态权重部分，避免重复加载基础模型权重"""
        try:
            import os
            if not os.path.exists(path):
                return False
            
            print(f"[BrainAI] 正在从 {path} 恢复学习成果...")
            state = torch.load(path, map_location=self.device, weights_only=False)
            
            # 新格式：dynamic_weights 字典
            if 'dynamic_weights' in state:
                dynamic_weights = state['dynamic_weights']
                restored_count = 0
                
                for name, param in self.model.model.named_parameters():
                    if name in dynamic_weights:
                        param.data.copy_(dynamic_weights[name])
                        restored_count += 1
                
                print(f"[BrainAI] [OK] 已恢复 {restored_count} 个动态权重层")
            
            # 兼容旧格式：model_state_dict
            elif 'model_state_dict' in state:
                model_state = state['model_state_dict']
                
                # 只加载动态权重层（避免覆盖静态权重）
                restored_count = 0
                for name, param in self.model.model.named_parameters():
                    if 'dynamic_weight' in name and name in model_state:
                        param.data.copy_(model_state[name])
                        restored_count += 1
                
                print(f"[BrainAI] [OK] 已恢复 {restored_count} 个动态权重层（兼容模式）")
            
            # 恢复其他状态
            if 'adapter_state_dict' in state:
                self.feature_adapter.load_state_dict(state['adapter_state_dict'])
            if 'adapter_optimizer_state_dict' in state:
                self.adapter_optimizer.load_state_dict(state['adapter_optimizer_state_dict'])
            
            # 重置推理引擎内部缓存组件
            for name, module in self.model.model.base_model.named_modules():
                if hasattr(module, '_cache_valid'):
                    module._cache_valid = False
            
            if 'hippocampus_state' in state:
                self.hippocampus.set_state(state['hippocampus_state'])
            if 'monologue_history' in state:
                self.monologue_history = state['monologue_history']
            if 'cycle_count' in state:
                self.cycle_count = state['cycle_count']
            if 'total_stdp_updates' in state:
                self.total_stdp_updates = state['total_stdp_updates']
            if 'current_thought_state' in state:
                self.current_thought_state = state['current_thought_state']
            
            print(f"[BrainAI] [OK] 学习成果恢复完成")
            return True
        except Exception as e:
            logger.error(f"动态权重加载失败: {e}。将重新初始化。")
            return False

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
    
    def generate_thought_stream(self, max_chunks: int = 5):
        """
        流式思维生成 - 高刷新小数据
        
        每次生成15-25个token，模拟人脑的思维流
        
        Args:
            max_chunks: 最大思维片段数
        
        Yields:
            dict: {'type': 'char', 'content': char} 或 {'type': 'chunk_end', 'content': full_chunk}
        """
        # 直接使用已验证的独白生成方法
        for _ in range(max_chunks):
            monologue = self._generate_spontaneous_monologue(max_tokens=25, temperature=0.75)
            # 清理独白内容
            monologue = self._clean_monologue_for_stream(monologue)
            
            if monologue and len(monologue) > 3:
                # 流式输出
                for char in monologue:
                    yield {'type': 'char', 'content': char}
                yield {'type': 'chunk_end', 'content': monologue}
            else:
                # 如果生成失败，使用预设的思维片段
                thoughts = [
                    "分析当前情况...",
                    "思考问题的本质...",
                    "推理可能的结论...",
                    "验证逻辑链条...",
                    "综合各种因素..."
                ]
                selected = random.choice(thoughts)
                for char in selected:
                    yield {'type': 'char', 'content': char}
                yield {'type': 'chunk_end', 'content': selected}
    
    def _clean_monologue_for_stream(self, text: str) -> str:
        """清理流式独白内容"""
        if not text:
            return ""
        
        # 移除特殊标签
        for tag in ['<|im_end|>', '<|im_start|>', '</system>', '<system>', '</user>', '<user>', '[', ']']:
            text = text.replace(tag, '')
        
        # 移除多余的点和数字（如"1.0.%"）
        import re
        text = re.sub(r'\d+\.\d+\.\d+\.?', '', text)
        text = re.sub(r'\.{3,}', '...', text)
        
        # 移除开头和结尾的空白
        text = text.strip()
        
        # 如果内容太短或无意义，返回空
        if len(text) < 2 or not any(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in text):
            return ""
        
        # 限制长度
        if len(text) > 60:
            # 在标点符号处截断
            for end_marker in ['。', '，', '！', '？', '...']:
                pos = text.rfind(end_marker, 10, 55)
                if pos > 0:
                    text = text[:pos+1]
                    break
            else:
                text = text[:50] + "..."
        
        return text
    
    def get_quick_response(self, user_input: str = "") -> str:
        """获取快速响应填充词"""
        if not self.inner_thought_engine:
            raise RuntimeError("内心思维独白引擎未初始化，无法生成快速响应")
        return self.inner_thought_engine.get_quick_response(user_input)
    
    def get_thought_flow_stats(self) -> dict:
        """获取思维流统计"""
        if not self.inner_thought_engine:
            raise RuntimeError("内心思维独白引擎未初始化，无法获取统计信息")
        return self.inner_thought_engine.get_stats()
