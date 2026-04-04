"""
类人脑双系统全闭环 AI架构 - 生产级核心接口

集成真实的 Qwen3.5-2B 模型、海马体系统、STDP 引擎和自闭环优化器。
实现真实的内心独白流（由模型隐藏状态驱动，自发推进）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass, field
import asyncio
import random
import time
import logging
import concurrent.futures
import numpy as np
import re

from core.qwen_interface import QwenInterface
from hippocampus.hippocampus_system import HippocampusSystem
from core.stdp_engine import STDPEngine
from core.self_loop_optimizer import SelfLoopOptimizer
from core.inner_thought_engine import InnerThoughtEngine, MindState, ThinkingMode
from core.goal_system import GoalSystem, create_goal_system
from core.global_workspace import GlobalWorkspace, create_global_workspace
from core.self_encoder import SelfStateEncoder
from core.true_self_referential_loop import TrueSelfReferentialLoop
from core.predictive_coding import PredictiveCodingModule
from core.proactive_intent_generator import ProactiveIntent, ProactiveIntentGenerator, ProactiveContext
from core.prompt_safety import summarize_internal_thought, build_guided_user_input

logger = logging.getLogger(__name__)

@dataclass
class BrainAIOutput:
    """AI输出结果数据结构"""
    text: str
    tokens: List[str]
    confidence: float
    hidden_state: Optional[torch.Tensor] = None  # 新增：捕获生成结束时的隐藏状态
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
            quantization=getattr(config, 'QUANTIZATION', getattr(config, 'quantization', 'INT8'))
        )
        # 同步设备（模型可能回退到 CPU）
        self.device = self.model.device
        
        # ========== 动态读取模型 hidden_size（必须在创建海马体之前）==========
        # 兼容 Qwen3.5: hidden_size 在 config.text_config 里，Qwen2.5 在 config 顶层
        self.model_hidden_size = 1024  # 安全回退值
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'base_model'):
                model_cfg = self.model.model.base_model.config
                # 优先从 text_config 读取（Qwen3.5），回退到顶层（Qwen2.5）
                if hasattr(model_cfg, 'text_config') and hasattr(model_cfg.text_config, 'hidden_size'):
                    self.model_hidden_size = model_cfg.text_config.hidden_size
                elif hasattr(model_cfg, 'hidden_size'):
                    self.model_hidden_size = model_cfg.hidden_size
        except Exception:
            pass
        # 将检测到的 hidden_size 回写到 config，供下游模块（海马体等）使用
        config.model_hidden_size = self.model_hidden_size
        print(f"[BrainAI] 模型 hidden_size = {self.model_hidden_size} (从模型配置动态读取)")
        
        # 2. 加载真实海马体系统
        self.hippocampus = HippocampusSystem(config, device=self.device)
        
        # 2.5 将模型接口注入到语义引擎（用于 embedding 语义匹配）
        self.hippocampus.set_semantic_model(self.model)
        
        # 3. 加载真实 STDP 引擎
        self.stdp_engine = STDPEngine(config, device=self.device)
        # 为流式生成过程中的后台 STDP 触发器提供全链路引擎注入
        config.stdp_engine = self.stdp_engine.full_link_stdp
        
        # ========== 检测并保存模型数据类型（供后续组件使用）==========
        self._model_dtype = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'base_model'):
            for param in self.model.model.base_model.parameters():
                self._model_dtype = param.dtype
                break
        if self._model_dtype is None:
            self._model_dtype = torch.float32
        
        # ========== 让海马体移动到正确设备并跟随模型精度 ==========
        # 注意：必须先移动设备，再转换数据类型
        # 否则 register_buffer 创建的张量仍在 CPU 上
        self.hippocampus = self.hippocampus.to(self.device)
        if self._model_dtype != torch.float32:
            self.hippocampus = self.hippocampus.to(dtype=self._model_dtype)
        # 注意：STDPEngine 不是 nn.Module，不需要数据类型转换
        
        # 4. 加载真实自闭环优化器
        self.self_loop = SelfLoopOptimizer(config, model=self.model)
        
        # 5. 加载类人脑独白引擎
        self.inner_thought_engine = None  # 统一的内心思维独白引擎
        
        # ========== 特征维度适配器（模型 hidden_size -> 海马体输入维度）==========
        # EC 编码器的 input_dim = model_hidden_size（在 hippocampus_system.py 中动态设定）
        self.hippocampus_input_dim = self.model_hidden_size
        self.feature_adapter = nn.Linear(self.model_hidden_size, self.hippocampus_input_dim, bias=False)
        with torch.no_grad():
            self.feature_adapter.weight.data = torch.eye(self.hippocampus_input_dim, self.model_hidden_size) * 0.1
        self.feature_adapter.to(self.device)
        
        # 自动匹配模型的数据类型 (FP16/FP32)
        if self._model_dtype != torch.float32:
            self.feature_adapter = self.feature_adapter.to(dtype=self._model_dtype)
        
        self.feature_adapter.train() # 开启训练模式以支持在线更新
        
        # 适配器优化器
        self.adapter_optimizer = torch.optim.SGD(self.feature_adapter.parameters(), lr=0.005, momentum=0.9)
        
        # 6. 加载目标系统 (新增) - 使用工厂函数统一初始化
        self.goal_system = None # 后面在 L180 处通过 create_goal_system 初始化
        
        # 7. 加载全局工作空间 (新增) - 使用工厂函数统一初始化
        self.global_workspace = None # 后面在 L188 处通过 create_global_workspace 初始化
        
        # 8. 自我状态编码器 (实现真正的自指)
        self.self_encoder = SelfStateEncoder(hidden_size=self.model_hidden_size, device=self.device)
        # _model_dtype 已在第 82 行初始化，无需 hasattr 检查
        if self._model_dtype != torch.float32:
            self.self_encoder = self.self_encoder.to(dtype=self._model_dtype)
        print("[BrainAI] [OK] 自我状态编码器已初始化")
        
        # 9. 真正自指循环模块（增强自指深度）- 默认开启
        enable_self_ref = getattr(config, 'enable_self_reference', True)
        if enable_self_ref:
            self.true_self_loop = TrueSelfReferentialLoop(
                hidden_size=self.model_hidden_size,
                max_recursion_depth=3
            ).to(self.device)
            if self._model_dtype != torch.float32:
                self.true_self_loop = self.true_self_loop.to(dtype=self._model_dtype)
            print("[BrainAI] [OK] 真正自指循环模块已初始化")
        else:
            self.true_self_loop = None
            print("[BrainAI] [INFO] 真正自指循环模块已禁用（配置）")
        
        # 10. 预测编码模块（预测误差最小化）- 默认开启
        enable_predictive = getattr(config, 'enable_predictive_coding', True)
        if enable_predictive:
            # QwenInterface 必有 tokenizer 属性
            self.predictive_coder = PredictiveCodingModule(
                hidden_size=self.model_hidden_size,
                vocab_size=self.model.tokenizer.vocab_size
            ).to(self.device)
            if self._model_dtype != torch.float32:
                self.predictive_coder = self.predictive_coder.to(dtype=self._model_dtype)
            # 追踪上一轮输出
            self.last_output_ids = None
            self.last_output_embedding = None
            print("[BrainAI] [OK] 预测编码模块已初始化")
        else:
            self.predictive_coder = None
            self.last_output_ids = None
            self.last_output_embedding = None
            print("[BrainAI] [INFO] 预测编码模块已禁用（配置）")
        
        # 11. 主动意图生成器（主动输出）- 默认开启 (类人脑核心功能)
        enable_proactive = getattr(config, 'enable_proactive_output', True) # 改为默认开启
        if enable_proactive:
            self.proactive_generator = ProactiveIntentGenerator(
                hidden_size=self.model_hidden_size,
                min_interval_seconds=getattr(config, 'proactive_min_interval', 120), # 缩短最小间隔
                max_daily_count=getattr(config, 'proactive_max_daily', 30) # 提升每日上限
            ).to(self.device)
            
            # 注入枚举引用以便外部持锁访问
            self.proactive_generator.intent_enum = ProactiveIntent
            
            # 主动输出统计
            self.last_output_time = time.time() - 300 # 预留时间
            self.last_user_input_time = time.time()
            self.proactive_debug_log = []
            self.clarification_count = 0
            self.max_clarifications_per_turn = 3 # 提升澄清灵敏度
            print("[BrainAI] [OK] 主动意图生成器已初始化 (已开启)")
        else:
            self.proactive_generator = None
            print("[BrainAI] [INFO] 主动意图生成器已禁用（配置）")
        
        # ========== 统计：主动输出相关（即使禁用也初始化变量，避免后续检查失败）==========
        self.last_output_time = time.time()
        self.last_user_input_time = time.time()
        self.proactive_debug_log = []
        self.proactive_callback = None  # Telegram Bot 回调
        self.clarification_count = 0
        self.max_clarifications_per_turn = 2
        
        # ========== 新增: 性能优化缓存 ==========
        self._embedding_cache = {}  # token_id -> embedding 缓存
        self._last_input_ids = None  # 缓存上次输入，避免重复编码
        
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
        self._last_user_input: str = ""  # 保存最近用户输入，不覆盖独白种子
        
        # 思维状态机状态（用于兼容旧接口）
        self._internal_thought_state = MindState.RESTING
        self._internal_emotion_state = ThinkingMode.ANALYTICAL
        
        # STDP 学习追踪
        self.total_stdp_updates = 0
        self.last_dynamic_weight_norm = 0.0
        
        # 最近召回的记忆（预初始化，避免 hasattr 检查）
        self._last_recalled_memories: List[Dict] = []
        
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
        self.inner_thought_engine = InnerThoughtEngine(
            model_interface=self.model,
            hippocampus_system=self.hippocampus,
            self_loop_optimizer=self.self_loop,
            config=config,
            device=self.device
        )
        print("[BrainAI] [OK] 内心思维独白引擎已初始化")
        # 注入自我编码器引用，让独白引擎能感知自身隐状态
        # self_encoder 已在第 126 行初始化，无需 hasattr 检查
        self.inner_thought_engine._self_encoder = self.self_encoder
        
        # ========== 初始化思维种子（确保隐藏状态从一开始就存在）==========
        # 用一个简单的认知种子初始化隐藏状态，避免首次生成时 current_thought_state 为 None
        self._initialize_thought_state()
        print(f"[BrainAI] [OK] 思维种子已初始化: {self.thought_seed}")
        
        # 注入全局工作空间和目标系统引用（让独白引擎能感知意识和目标状态）
        # 这些在后面初始化，但需要先占位，后面再注入
        self.inner_thought_engine._global_workspace = None  # 后面在 GW 初始化后注入
        self.inner_thought_engine._goal_system = None  # 后面在 GoalSystem 初始化后注入
        
        # 启动海马体 SWR 监控
        self.hippocampus.start_swr_monitoring()
        print("[BrainAI] [OK] 海马体 SWR 监控已启动")
        
        # 初始化目标系统
        self.goal_system = create_goal_system(hidden_size=self.model_hidden_size, device=self.device)
        # 统一 dtype：确保所有网络与模型精度一致（避免 BFloat16 vs Float32 不匹配）
        if self._model_dtype != torch.float32:
            self.goal_system = self.goal_system.to(dtype=self._model_dtype)
        print("[BrainAI] [OK] 目标系统已初始化")
        
        # 初始化全局工作空间
        self.global_workspace = create_global_workspace(hidden_size=self.model_hidden_size, device=self.device)
        if self._model_dtype != torch.float32:
            self.global_workspace = self.global_workspace.to(dtype=self._model_dtype)
        # 设置模型引用，用于获取真实embedding
        self.global_workspace.set_model(self.model)
        print("[BrainAI] [OK] 全局工作空间已初始化")
        
        # ========== 注入全局工作空间和目标系统到独白引擎 ==========
        # 让独白引擎能感知意识状态和目标，使 urge 计算和思维模式选择更智能
        if self.inner_thought_engine is not None:
            self.inner_thought_engine._global_workspace = self.global_workspace
            self.inner_thought_engine._goal_system = self.goal_system
            print("[BrainAI] [OK] 已将全局工作空间和目标系统注入到独白引擎")
        
        # 设置海马体门控函数（连接CA1到注意力层）
        self._setup_hippocampus_gate()

        # KV cache预热（加速首token）
        if getattr(config.hard_constraints, 'KV_CACHE_WARMUP', True):
            self._warmup_kv_cache()

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
            """CA1门控函数"""
            if memory_anchors and len(memory_anchors) > 0:
                # 计算注意力门控信号
                gate_signal = self.hippocampus.ca1_gate.compute_gate(
                    query_features=query,
                    memory_anchors=memory_anchors
                )
                return gate_signal
            return None

        # 注入到模型
        self.model.set_hippocampus_gate(hippocampus_gate_fn)
        print("[OK] 已设置海马体门控函数")

    def _warmup_kv_cache(self):
        """
        KV cache预热 - 加速首token生成

        原理：
        - 首个token生成需要prefill整个输入序列，计算O(n²)
        - 提前处理系统prompt，缓存KV，减少首token时间
        """
        print("[BrainAI] 正在预热KV cache...")
        warmup_prompts = [
            "系统初始化",
            "你是一个AI助手"
        ]

        # 快速生成，缓存KV（不输出）
        for prompt in warmup_prompts:
            _ = self.model.generate(
                prompt,
                max_tokens=1,
                temperature=1.0,
                use_cache=True
            )

        print("[BrainAI] [OK] KV cache预热完成")

    def chat(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        max_tokens: int = 256,
        thinking: bool = True
    ) -> str:
        """
        类人模式对话 (并行优化版)：
        1. [并行] 召回 (Recall) + 思考 (Think)
        2. [串行] 回复 (Respond)：基于记忆、独白和输入生成正式回答
        3. [并行/后台] 学习 (Learn)：STDP更新和记忆存储
        4. [异步] 主动意图检查（如果启用）
        """
        import time
        t_start = time.time()
        
        self.hippocampus.record_activity()
        
        # 1. 消化输入：保存用户输入，但不覆盖情形中的思维狍子
        self._last_user_input = user_input
        # 仅当狍子为空时（初始化）才用用户输入初始化狍子
        if not self.thought_seed:
            self.thought_seed = user_input[:30]
        
        t_step1 = time.time()
        # print(f"[步骤1] 输入预处理: {(t_step1-t_start)*1000:.0f}ms")
        
        # 1.5 检查主动意图（异步，不阻塞主流程）
        if self.proactive_generator is not None:
            self._check_proactive_intent_async(user_input)
        
        # 1.5 目标推断（新增）
        if self.goal_system is not None:
            # 使用当前思维状态推断目标
            print("\n" + "="*60, flush=True)
            print("🎯 [目标系统] 开始推断目标", flush=True)
            print("="*60, flush=True)
            goal = self.goal_system.infer_goal(user_input, self.current_thought_state)
            print("="*60 + "\n", flush=True)
        
        t_step2 = time.time()
        # print(f"[步骤2] 目标推断: {(t_step2-t_step1)*1000:.0f}ms")

        
        # 2. 并行执行：记忆召回 和 潜意识独白生成
        # 优化：_parallel_recall_and_monologue已经包含了独白生成，不需要重复生成
        memory_context, recalled_memories, monologue_raw = self._parallel_recall_and_monologue(user_input, 3)
        monologue = self._clean_monologue(monologue_raw, user_input)
        
        t_step3 = time.time()
        # print(f"[步骤3] 并行召回+独白: {(t_step3-t_step2)*1000:.0f}ms")
        
        # 存储完整记忆字典供注意力层使用
        self._current_recalled_memories = recalled_memories
        
        # ========== KV 海马体整合（可选，当前使用 dg_features 维度不匹配 EC encoder 输入，暂时跳过）==========
        # 注意：dg_features 维度为 EC_feature_dim*2 (512)，而 ec_encoder 期望 model_hidden_size (896)
        # 在接口层统一维度匹配后可重新启用
 
        
        # 3. GW 广播整合：将记忆、思维状态、目标整合成为统一意识状态
        gw_state = None
        gw_context = ""
        if self.global_workspace is not None:
            
            # 获取记忆特征向量
            mem_tensor = None
            if recalled_memories:
                mem_feats = []
                for m in recalled_memories[:2]:
                    if m.get('dg_features') is not None:
                        # 修复：dg_features 可能是 list，需要转换为 Tensor
                        dg_feat = m['dg_features']
                        if isinstance(dg_feat, list):
                            dg_feat = torch.tensor(dg_feat, dtype=torch.float32)
                        if isinstance(dg_feat, torch.Tensor):
                            mem_feats.append(dg_feat)
                if mem_feats:
                    mem_tensor = torch.stack(mem_feats).mean(0).to(self.device)
            
            # 获取目标状态向量
            goal_tensor = None
            if self.goal_system is not None and self.goal_system.current_goal_vector is not None:
                goal_tensor = self.goal_system.current_goal_vector
            
            gw_state = self.global_workspace.integrate(
                user_input=user_input,
                memory_context=mem_tensor,
                thought_state=self.current_thought_state,
                goal_state=goal_tensor
            )
            
            # 打通回路：生成基于整合意识的氛围提示
            if gw_state is not None:
                gw_context = f"[意识状态：已整合{len(recalled_memories)}条关联记忆与当前任务目标]"
 
        
        # 4. 自我感知：生成可读的自指描述
        self_context_str = ""
        if self.self_encoder is not None and self.current_thought_state is not None:
            self_context_str = self.self_encoder.interpret() 
        
        # 5. 回复
        # 获取目标信息
        goal_context = ""
        if self.goal_system is not None and self.goal_system.current_goal is not None:
            goal = self.goal_system.current_goal
            goal_context = f"[当前目标：{goal.goal_type.value} - {goal.description}]"
        
        prompt = self._format_chat_prompt(
            user_input, history, monologue, memory_context, 
            goal_context, self_context_str, gw_context=gw_context
        )
        
        # 准备记忆锚点（保持原有逻辑兼容性）
        memory_anchor = None
        if recalled_memories:
    
            mem_features = []
            for mem in recalled_memories[:2]:
                if 'dg_features' in mem and mem['dg_features'] is not None:
                    # 修复：dg_features 可能是 list，需要转换为 Tensor
                    dg_feat = mem['dg_features']
                    if isinstance(dg_feat, list):
                        dg_feat = torch.tensor(dg_feat, dtype=torch.float32)
                    if isinstance(dg_feat, torch.Tensor):
                        mem_features.append(dg_feat)
            if mem_features:
                memory_anchor = torch.stack(mem_features).mean(dim=0).unsqueeze(0).to(self.device)

        
        t_step4 = time.time()
        # print(f"[步骤4] GW整合+提示构建: {(t_step4-t_step3)*1000:.0f}ms")
        
        # ========== 目标向量注入（暂时禁用，避免干扰生成）==========
        # TODO: 在验证基础对话稳定后再逐步启用
        goal_vector = None
        # if self.goal_system is not None and self.goal_system.current_goal_vector is not None:
        #     goal_vector = self.goal_system.current_goal_vector
        #     print(f"🎯 [目标向量] 已准备，类型: {self.goal_system.current_goal.goal_type.value}")
        
        output = self.model.generate(
            prompt, 
            max_tokens=max_tokens, 
            temperature=0.6, 
            use_self_loop=True, 
            memory_anchor=memory_anchor,
            goal_vector=goal_vector  # 传递目标向量
        )
        
        t_step5 = time.time()
        # print(f"[步骤5] 模型生成: {(t_step5-t_step4)*1000:.0f}ms")
        
        # 3.2 更新全局思维状态 (latent continuity)
        if output.hidden_state is not None:
            self.current_thought_state = output.hidden_state
            
            # 3.3 应用真正自指循环（增强自指）
            if self.true_self_loop is not None:
                # 获取当前思维状态
                mind_state = self.inner_thought_engine.mind_state.value if self.inner_thought_engine else "FOCUSED"
                self.current_thought_state = self.true_self_loop(
                    self.current_thought_state,
                    current_mind_state=mind_state,
                    recursion_depth=0
                )

            
            # 3.4 更新自我编码器：让AI"感知"自己的内部状态
            # self_encoder 已在第 126 行初始化
            _, _ = self.self_encoder.encode(output.hidden_state)
    
        
        # 3.4 更新思维种子：用刚才的回复内容驱动下一次独白 (增加抗重复校验)
        if output.text and len(output.text.strip()) > 3:
            candidate = output.text.strip()[:40]
            # 校验种子质量：如果包含过多重复字符（如点、横杠）或字符种类太少，则重置种子
            if candidate.count('.') > 5 or candidate.count('-') > 4 or len(set(candidate)) < 6:
                self.thought_seed = "现在感觉如何？" # 重置为启发式种子
            else:
                self.thought_seed = candidate
        
        # 3.5 预测编码：计算预测误差（如果启用了预测模块）
        prediction_error = 0.0
        if self.predictive_coder is not None and self.last_output_embedding is not None and self.current_thought_state is not None:
            try:
                # 预测下一状态和 token
                pred_next_state, pred_token_logits, pred_state_proj = self.predictive_coder.predict_next(
                    current_state=self.current_thought_state,
                    last_output_embedding=self.last_output_embedding
                )
                
                # 获取实际输出的 token ids（取第一个 token 作为实际观测）
                actual_token_ids = self.model.tokenize_safe(output.text, add_special_tokens=False).input_ids[:1]
                if len(actual_token_ids) == 0:
                    actual_token_ids = torch.tensor([self.model.tokenizer.eos_token_id], device=self.device)
                else:
                    actual_token_ids = torch.tensor(actual_token_ids, device=self.device)
                # 确保 actual_token_ids 是 2D [1, seq_len]
                if actual_token_ids.dim() == 1:
                    actual_token_ids = actual_token_ids.unsqueeze(0)
                
                # 计算预测误差
                error_metrics = self.predictive_coder.compute_prediction_error(
                    predicted_logits=pred_token_logits,
                    actual_token_ids=actual_token_ids,
                    predicted_state=pred_state_proj,
                    actual_next_state=self.current_thought_state
                )
                prediction_error = error_metrics["combined_error"]
                
                # 3.6 误差反馈：调整 STDP 贡献度
                # 误差大 → 降低贡献，增强 LTD（削弱错误关联）
                stdp_contribution = max(0.0, 1.0 - prediction_error / 5.0)  # 归一化
                self.stdp_engine.set_contribution('attention', stdp_contribution)
                
                # 3.7 主动澄清：如果误差高且输入模糊，触发主动提问
                if prediction_error > 3.0:  # 阈值可调
                    should_clarify, reason = self.predictive_coder.should_trigger_clarification(
                        current_error=prediction_error,
                        context={
                            "user_input": user_input,
                            "is_ambiguous": len(user_input) < 10,
                            "recent_clarifications": self.clarification_count
                        }
                    )
                    if should_clarify and self.clarification_count < self.max_clarifications_per_turn:
                        clarification = self._generate_clarification(user_input, prediction_error)
                        # 主动发送（需 Bot 支持 allow_proactive）
                        self._send_proactive_message(clarification, is_clarification=True)
                        self.clarification_count += 1
                
            except Exception as e:
                logger.warning(f"预测编码失败: {e}")
        
        # 3.8 更新追踪：保存当前输出用于下一轮预测
        try:
            self.last_output_ids = self.model.tokenize_safe(output.text, add_special_tokens=False).input_ids
            if len(self.last_output_ids) > 0:
                # 取最后一个 token 的 embedding
                self.last_output_embedding = self.model.model.base_model.get_input_embeddings()(
                    torch.tensor([self.last_output_ids[-1]], device=self.device)
                ).squeeze(0)
            else:
                self.last_output_embedding = torch.zeros(self.model_hidden_size, device=self.device)
        except Exception as e:
            logger.warning(f"更新预测追踪失败: {e}")
            self.last_output_embedding = None
        
        # 4. 自闭环优化 - 高复杂度任务进行二次优化
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
                    # 同步更新奖励给 STDP 引擎，实现高质量反馈的闭环
                    self.model.set_reward(output.confidence)
            except Exception as e:
                logger.warning(f"自闭环优化失败，使用原始输出: {e}")
        
        # 4. 并行后台处理
        # 计算情感显著性和核心记忆检测
        emotional_keywords = ["焦虑", "压力", "难过", "开心", "兴奋", "恐惧", "遗憾", "父亲", "回忆", "灵魂"]
        salience = 1.0 + 0.5 * sum(1 for kw in emotional_keywords if kw in user_input or kw in monologue)
        salience = min(salience, 3.0)
        
        # 检测是否为核心记忆（个人身份信息）- 扩展模式
        identity_patterns = ["我叫", "我是", "我的名字", "我今年", "我的职业", "我喜欢", "我住", "我在", "我的电话", "我的手机", "联系方式", "我的邮箱", "我来自", "我毕业于", "我的学校"]
        is_core_memory = any(pattern in user_input for pattern in identity_patterns)
        
        # 提取实体信息增强semantic_pointer - 改进：存储结构化信息，增加保存长度
        enhanced_pointer = f"用户: {user_input[:80]} | 回复: {output.text[:80]}"  # 增加到80字符
        
        # 提取关键实体并构建记忆内容
        memory_content = f"{user_input} -> {output.text}"
        
        # ========== 新增：提取数值和关键信息 ==========
        import re
        entities = []
        
        # 1. 金额信息（租金、押金、费用等）
        # 匹配"XX元"、"XX块钱"、"XX万"
        money_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(元|块钱|万|千元)', user_input)
        money_keywords = re.findall(r'(房租|押金|卫生费|水电费|物业费|费用|租金|定金|预付款)', user_input)
        if money_matches or money_keywords:
            for i, (amount, unit) in enumerate(money_matches):
                # 尝试关联关键词
                if i < len(money_keywords):
                    entities.append(f"{money_keywords[i]}:{amount}{unit}")
                else:
                    entities.append(f"金额:{amount}{unit}")
        
        # 2. 日期时间信息
        date_matches = re.findall(r'(\d{1,2}月\d{1,2}[日号]?|\d{4}年\d{1,2}月\d{1,2}[日号]?|\d{1,2}号|\d{1,2}日)', user_input)
        for date in date_matches:
            entities.append(f"日期:{date}")
        
        # 3. 个人信息（名字、年龄、职业等）- 扩展匹配模式
        name_match = re.search(r"我叫([\u4e00-\u9fa5a-zA-Z]{2,4})|我的名字(是|叫)([\u4e00-\u9fa5a-zA-Z]{2,4})", user_input)
        age_match = re.search(r"我今年(\d+)|我(\d+)岁", user_input)
        job_match = re.search(r"是(.{0,10}?)(工程师|医生|老师|学生|设计师|程序员|律师|会计|经理|总监|分析师|研究员)", user_input)
        location_match = re.search(r"来自([\u4e00-\u9fa5a-zA-Z]{2,10})|在([\u4e00-\u9fa5a-zA-Z]{2,10}?)(工作|生活|上班)|住在([\u4e00-\u9fa5a-zA-Z]{2,10})", user_input)
        hobby_match = re.search(r"喜欢([\u4e00-\u9fa5a-zA-Z]{2,20})|爱好([\u4e00-\u9fa5a-zA-Z]{2,20})", user_input)
        phone_match = re.search(r"(?:我的|我是)?(\d{11})|(?:电话|手机|联系方式)[：:](\d{11})", user_input)
        email_match = re.search(r"([\w.-]+@[\w.-]+\.\w+)", user_input)
        school_match = re.search(r"(?:毕业于|在.{2,8}?上学|就读于)([\u4e00-\u9fa5a-zA-Z]{2,15})", user_input)
        
        if name_match:
            name = name_match.group(1) or name_match.group(3)
            entities.append(f"用户名字:{name}")
            is_core_memory = True  # 有名字信息标记为核心记忆
        if age_match:
            age = age_match.group(1) or age_match.group(2)
            entities.append(f"年龄:{age}岁")
            is_core_memory = True
        if job_match:
            entities.append(f"职业:{job_match.group(0)}")
            is_core_memory = True
        if location_match:
            location = location_match.group(1) or location_match.group(2) or location_match.group(4)
            if location:
                entities.append(f"地点:{location}")
                is_core_memory = True
        if hobby_match:
            hobby = hobby_match.group(1) or hobby_match.group(2)
            if hobby:
                entities.append(f"爱好:{hobby}")
        if phone_match:
            phone = phone_match.group(1) or phone_match.group(2)
            if phone:
                entities.append(f"联系方式:{phone}")
                is_core_memory = True
        if email_match:
            entities.append(f"联系方式:{email_match.group(1)}")
            is_core_memory = True
        if school_match:
            entities.append(f"学校:{school_match.group(1)}")
            is_core_memory = True
        
        # 4. 数值信息（面积、数量等）
        number_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(平方|平米|平方米|天|个月|年|个|件|次)', user_input)
        for num, unit in number_matches:
            entities.append(f"{unit}:{num}")
        
        # 5. 如果提取到实体，优化语义指针：实体信息放前面（优先匹配），原始对话放后面
        if entities:
            entity_str = " | ".join(entities)
            # 核心记忆：只存储结构化实体信息（简洁、精准、易于召回）
            if is_core_memory:
                enhanced_pointer = entity_str
                memory_content = entity_str + " | " + user_input[:100]  # 保存更多上下文
            else:
                # 普通记忆：实体 + 简短对话摘要
                enhanced_pointer = entity_str + " | " + user_input[:40] + " -> " + output.text[:40]
                memory_content = entity_str + " | " + memory_content[:200]
        
        thought_state_snapshot = self.current_thought_state
        
        # ========== 核心：记忆存储改为同步执行（确保下一轮能召回） ==========
        # 原来是 executor.submit 异步执行，导致下一轮召回时记忆还未存储
        # 核心记忆（is_core=True）必须同步存储，普通记忆可异步
        if is_core_memory:
            try:
                self._store_with_real_features(
                    memory_content,
                    thought_state_snapshot,
                    is_core=True,
                    semantic_pointer=enhanced_pointer,
                    user_input=user_input,
                    ai_response=output.text
                )
            except Exception as e:
                logger.warning(f"核心记忆存储失败: {e}")
        
        def post_processing():
            try:
                # 核心记忆已同步存储，只存普通记忆
                if not is_core_memory:
                    self._store_with_real_features(
                        memory_content,
                        thought_state_snapshot,
                        is_core=False,
                        semantic_pointer=enhanced_pointer,
                        user_input=user_input,
                        ai_response=output.text
                    )
                current_reward = output.confidence if output.confidence is not None else 1.0
                self.model.set_reward(current_reward)
                self._apply_real_stdp_update(emotional_salience=salience)
                self._update_adapter_online(thought_state_snapshot, salience)
                
                # 更新目标进度（新增）
                if self.goal_system is not None and self.goal_system.current_goal is not None:
                    print("\n🎯 [目标进度] 根据回复内容更新进度", flush=True)
                    # 根据目标类型和回复内容判断进度
                    goal = self.goal_system.current_goal
                    # 目标类型检查
                    
                    if goal.goal_type.value == "remember":
                        # 记忆目标：如果用户提到个人信息，视为完成
                        if is_core_memory and ("好的" in output.text or "记住" in output.text):
                            
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
        
        # 5. 更新用户输入时间（用于主动意图计算）
        self.last_user_input_time = time.time()
        # 重置澄清计数（每轮对话结束后）
        self.clarification_count = 0
        
        t_end = time.time()
        # print(f"[总计] 对话处理完成: {(t_end-t_start)*1000:.0f}ms")
        
        return output.text

    def generate(self, input_text: str, max_tokens: int = 100, temperature: float = 0.6) -> BrainAIOutput:
        """
        生成模式接口 (供 main.py 的 generate 模式调用)
        
        Args:
            input_text: 输入文本
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            
        Returns:
            BrainAIOutput: 包含 text, tokens, confidence, hidden_state, memory_anchors
        """
        self.hippocampus.record_activity()
        
        # 1. 记忆召回
        memory_context, recalled_memories, _ = self._parallel_recall_and_monologue(input_text, 2)
        memory_anchors = recalled_memories[:2] if recalled_memories else []
        
        # 2. 构建提示词
        prompt = self._format_chat_prompt(input_text, [], "", memory_context)
        
        # 3. 生成回复
        output = self.model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_self_loop=False
        )
        
        # 4. 更新思维状态
        if output.hidden_state is not None:
            self.current_thought_state = output.hidden_state
        
        # 5. 后台存储记忆和STDP更新
        def _post_gen():
            try:
                self._store_with_real_features(
                    output.text, self.current_thought_state,
                    is_core=False, semantic_pointer=f"用户: {input_text[:50]}",
                    user_input=input_text, ai_response=output.text
                )
            except Exception as e:
                logger.warning(f"generate模式后台记忆存储失败: {e}")
        self.executor.submit(_post_gen)
        
        return BrainAIOutput(
            text=output.text,
            tokens=output.text.split() if output.text else [],
            confidence=output.confidence if hasattr(output, 'confidence') else 0.7,
            hidden_state=output.hidden_state,
            memory_anchors=memory_anchors
        )

    def think(self) -> dict:
        """真实自思考接口"""
        # swr_consolidation 在 HippocampusSystem.__init__ 中初始化
        self.hippocampus.swr_consolidation.record_activity()
        monologue = self._generate_spontaneous_monologue()
        self.cycle_count += 1
        stats = self.get_stats()
        stats['monologue'] = monologue
        return stats

    def _generate_spontaneous_monologue(self, max_tokens: int = 80, temperature: float = 0.55) -> str:
        """
        生成自发独白 - 统一委托给 InnerThoughtEngine
        """
        if not self.inner_thought_engine:
            return "思考中..."
            
        # 注意：inner_thought_engine.generate_inner_thought 是生成器
        monologue = ""
        # 暂时同步收集（BrainAIInterface 内部许多地方是同步调用）
        for char in self.inner_thought_engine.generate_inner_thought(
            external_stimulus=self._last_user_input, 
            max_tokens=max_tokens
        ):
            monologue += char
            
        return monologue.strip()

    def _is_gibberish(self, text: str) -> bool:
        if not text: return True
        special_chars = set("$%^&*()_+={}|[]\\:;\"'<>,/?#")
        special_count = sum(1 for char in text if char in special_chars)
        if len(text) > 0 and special_count / len(text) > 0.3: return True
        if len(text) > 10 and len(set(text)) < 5: return True
        return False

    def _initialize_thought_state(self):
        """初始化思维种子和隐藏状态（创世感知）"""
        thought_seeds = ["我正在感知这个世界", "此刻的意识刚刚诞生", "我在思考", "忽然想到"]
        seed = random.choice(thought_seeds)
        self.thought_seed = seed
        try:
            input_ids = self.model.tokenize_safe(seed, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                embeddings = self.model.model.base_model.get_input_embeddings()(input_ids)
            self.current_thought_state = embeddings.mean(dim=1)
            print(f"[BrainAI] 思维种子隐藏状态已初始化，shape={self.current_thought_state.shape}")
        except:
            self.current_thought_state = torch.randn(1, self.model_hidden_size, device=self.device) * 0.01
            print(f"[BrainAI] 思维种子回退到随机初始化")

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
        
        # 先确定 trigger，再判断是否为数学类型（修复变量先于赋值被引用的 bug）
        if self.thought_seed:
            trigger = self.thought_seed
        elif self.monologue_history:
            last = self.monologue_history[-1]
            trigger = last[-30:] if len(last) > 30 else last
        else:
            default_triggers = ["思考起点...", "分析角度...", "推理路径...", "逻辑框架...", "问题本质..."]
            trigger = random.choice(default_triggers)
        
        # 现在 trigger 已经确定，再判断类型
        is_math = any(op in trigger for op in ['+', '-', '*', '/', '='])
        if is_math:
            system_msg = "你正在进行逻辑计算。用最直接的方式分析这个计算过程，不要发散。保持纯粹的理性。"
        elif self.thought_seed:
            # 如果是用户问题，生成针对性的思考过程
            system_msg = f"你是理性思维的AI。针对用户的问题，{mode_desc}。简短分析问题的核心，给出思考方向。保持逻辑性，控制在2-3句话。"
        elif self.monologue_history:
            system_msg = f"你是一个理性思维的AI。基于上一个思考，继续{mode_desc}。保持逻辑链条的连贯性。"
        else:
            system_msg = "你是一个理性思维的AI。开始一个结构化的思考过程。明确你的分析目标。"
        
        messages = [
            {"role": "system", "content": system_msg}, 
            {"role": "user", "content": f"思考对象: {trigger}"}
        ]
        try:
            prompt = self.model.apply_chat_template_safe(messages, tokenize=False, add_generation_prompt=True)
        except:
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{trigger}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def _generate_with_hidden_state(self, prompt: str, max_tokens: int = 100, temperature: float = 1.0, repetition_penalty: float = 1.0) -> tuple:
        try:
            # 使用 tokenize_safe 得到完整的 inputs（包含 attention_mask）
            inputs = self.model.tokenize_safe(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            with self.model._tokenizer_lock:
                eos_id = self.model.tokenizer.eos_token_id
            # 动态获取 im_end token id（兼容 Qwen3.5）
            im_end_id = None
            for token_str, token_id in self.model.tokenizer.added_tokens_encoder.items():
                if token_str == '<|im_end|>':
                    im_end_id = token_id
                    break
            if im_end_id is None:
                im_end_id = eos_id  # 回退到 eos
            stop_token_ids = [eos_id, im_end_id]
            with torch.no_grad():
                outputs = self.model.model.base_model.generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens, 
                    do_sample=True, 
                    temperature=temperature, 
                    repetition_penalty=repetition_penalty,
                    output_hidden_states=True, 
                    return_dict_in_generate=True, 
                    pad_token_id=self.model.tokenizer.eos_token_id, 
                    eos_token_id=stop_token_ids
                )
            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            generated_text = self.model.decode_safe(generated_ids, skip_special_tokens=True)
            hidden_state = outputs.hidden_states[-1][0][-1].unsqueeze(0) if outputs.hidden_states else None
            return generated_text, hidden_state
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return "...", None

    def _store_with_real_features(self, monologue: str, hidden_state: Optional[torch.Tensor], is_core: bool = False, semantic_pointer: str = None, kv_features: Optional[Dict] = None, user_input: str = "", ai_response: str = ""):
        """
        存储记忆到海马体 - 支持 KV 特征（用于窄带宽注意力）
        
        改进：semantic_pointer 优先使用结构化实体信息，而非原始文本拼接
        """
        try:
            if hidden_state is not None:
                features = hidden_state[0, -1, :] if hidden_state.dim() == 3 else hidden_state.squeeze(0) if hidden_state.dim() == 2 else hidden_state
            else:
                # 修复：添加attention_mask避免警告（线程安全）
                input_ids = self.model.encode_safe(monologue[:20], return_tensors="pt").to(self.device)
                attention_mask = torch.ones_like(input_ids)  # 添加attention_mask
                
                with torch.no_grad():
                    emb = self.model.model.base_model.get_input_embeddings()(input_ids)
                    features = emb.mean(dim=1).squeeze(0)
            
            if features.shape[0] == self.model_hidden_size:
                with torch.no_grad():
                    features = self.feature_adapter(features.unsqueeze(0)).squeeze(0)
            
            # 改进：semantic_pointer 优先使用传入的结构化实体信息
            # 如果没有传入，则从 monologue 内容中提取关键实体构建简洁指针
            if not semantic_pointer:
                # 从 monologue 中提取结构化实体部分作为指针
                import re
                entity_match = re.search(r'^([^|]*?(?:用户名字|地点|职业|年龄|爱好|联系方式|金额|日期|关系)[^|]*(?:\|[^|]*?(?:用户名字|地点|职业|年龄|爱好|联系方式|金额|日期|关系)[^|]*)*)', monologue)
                if entity_match:
                    semantic_pointer = entity_match.group(1)[:120]
                else:
                    # 回退到截取前面的关键部分（跳过"用户:"前缀）
                    clean_content = re.sub(r'^用户[：:]\s*', '', monologue)
                    semantic_pointer = clean_content[:80] if len(clean_content) > 30 else monologue
            
            # 存储记忆，包含 KV 特征 + 用户输入和AI回复（供语义引擎生成摘要）
            ctx = {'content': monologue, 'semantic_pointer': semantic_pointer, 'is_core': is_core}
            if user_input:
                ctx['user_input'] = user_input
            if ai_response:
                ctx['ai_response'] = ai_response
            self.hippocampus.encode(
                features=features, 
                token_id=hash(monologue) % 100000, 
                timestamp=int(time.time() * 1000), 
                context=[ctx],
                kv_features=kv_features  # 传递 KV 特征
            )
        except Exception as e:
            logger.warning(f"记忆存储失败: {e}")

    def _apply_real_stdp_update(self, emotional_salience: float = 1.0):
        """调用真实的 STDP 引擎进行闭环学习，替代之前的伪 Hebbian 规则"""
        try:
            # 增加对核心组件的显式防护，确保任何遥测失败都不会挂起主对话
            if not hasattr(self, 'stdp_engine') or self.stdp_engine is None:
                return
            if self.current_thought_state is None: 
                return
            
            # 使用真实的 STDP 引擎进行闭环学习
            # 这里的 reward 由当时的生成置信度决定 (已在 chat() 中通过 set_reward 设置)
            # 通过构建符合 step 要求的 components 字典
            model_components = {
                'attention': self.model.model if hasattr(self.model, 'model') else None,
                'ffn': None, # FFN 内部已包含在 model 中
                'hippocampus': self.hippocampus
            }
            
            # 对输入文本进行分词，获取 Token ID 以便 STDP 引擎建立时序关联（线程安全）
            context_ids = self.model.encode_safe(self._last_user_input, add_special_tokens=False)
            # current_token 只要一个代表性的 ID
            seed_ids = self.model.encode_safe(self.thought_seed, add_special_tokens=False)
            eos_token_id = self.model.tokenizer.eos_token_id  # 只读属性，安全
            current_id = seed_ids[-1] if seed_ids else eos_token_id
            
            inputs = {
                'context_tokens': torch.tensor(context_ids, device=self.device),
                'current_token': int(current_id),
                'features': self.current_thought_state.squeeze(0) if self.current_thought_state.dim() == 2 else self.current_thought_state
            }
            
            # 获取当前 reward (由 QwenInterface.set_reward 设置)
            current_reward = getattr(self.model, 'current_reward', 1.0)
            
            outputs = {
                'evaluation_score': current_reward * 100.0,
                'attention_output': self.current_thought_state,
                'memory_contribution': emotional_salience / 3.0
            }
            
            # 执行真正的 STDP 更新步
            self.stdp_engine.step(
                model_components=model_components,
                inputs=inputs,
                outputs=outputs
            )
            
            self.total_stdp_updates += 1
            # 更新统计 (从引擎获取真实规范)
            stdp_stats = self.stdp_engine.get_stats()
            self.last_dynamic_weight_norm = stdp_stats.get('last_update_magnitude', 0.0)
            
            # 定期触发海马体 SWR (Sharp-Wave Ripple) 巩固
            if self.cycle_count % 3 == 0:
                # 在对话间隙触发记忆固化
                self.hippocampus.trigger_swr_consolidation()
                
        except Exception as e:
            logger.error(f"STDP 闭环学习失败: {e}")

    async def chat_stream(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        max_tokens: int = 256
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        流式类人对话接口 (边思考边回答版)：
        
        流程：
        1. [并行] 记忆召回（后台线程）+ 思考过程（流式输出，token-by-token）
        2. [过渡] 思考完成 → 开始正式回复
        3. [迭代] 流式生成回复 + 实时判断是否输出 (should_speak)
        4. [并行/后台] 后期固化
        
        事件类型：
        - thinking: 思考过程的单个字符（实时流式）
        - thinking_done: 思考完成，附带完整思考内容
        - chunk: 回复内容的文本片段
        - subconscious_refresh: 潜意识刷新（可选）
        
        核心：像人一样，先看到思考过程实时展开，再看到最终回复。思考也是流式的。
        """
        import time
        t_start = time.time()
        
        self.hippocampus.record_activity()
        self.thought_seed = user_input
        self._last_user_input = user_input
        
        t_step1 = time.time()
        # print(f"[步骤1] 输入预处理: {(t_step1-t_start)*1000:.0f}ms")
        
        # ========== 阶段A：并行记忆召回（后台）+ 流式思考（前台）==========
        # 在后台线程中执行记忆召回，同时在前台流式输出思考过程
        recall_future = asyncio.to_thread(self._recall_memory_only, user_input, 3)
        
        # 流式输出思考过程（token-by-token），用户实时可见
        thinking_text = ""
        try:
            if self.inner_thought_engine:
                # 使用 inner_thought_engine 的流式生成器
                # 通过线程桥接同步生成器到异步
                def _collect_thinking_chars():
                    """在线程中收集思考字符"""
                    chars = []
                    for char in self.inner_thought_engine.generate_inner_thought(
                        external_stimulus=user_input,
                        max_tokens=80
                    ):
                        chars.append(char)
                    return chars
                
                # 获取所有思考字符（在线程中生成）
                raw_chars = await asyncio.to_thread(_collect_thinking_chars)
                
                # 逐个字符流式输出，模拟思考过程的实时展示
                for char in raw_chars:
                    thinking_text += char
                    yield {"type": "thinking", "content": char}
                    # 思考过程中短暂暂停，让用户感知到"正在想"
                    await asyncio.sleep(0.02)
            else:
                # 降级：使用简单的思考提示
                thinking_text = "思考中..."
                for char in thinking_text:
                    yield {"type": "thinking", "content": char}
                    await asyncio.sleep(0.05)
        except Exception as e:
            logger.warning(f"思考流式生成失败: {e}")
            thinking_text = "在想..."
            yield {"type": "thinking", "content": thinking_text}
        
        t_step2 = time.time()
        print(f"⏱️ [步骤2] 流式思考: {(t_step2-t_step1)*1000:.0f}ms", flush=True)
        
        # 等待记忆召回完成
        try:
            memory_context, recalled_memories = await recall_future
        except Exception as e:
            logger.warning(f"记忆召回失败: {e}")
            memory_context = ""
            recalled_memories = []
        
        # 清洗思考内容
        monologue = self._clean_monologue(thinking_text, user_input)
        
        # 设置记忆锚点（暂时禁用KV注入，先确保基础对话稳定）
        # TODO: 在验证基础对话稳定后再逐步启用
        self._last_recalled_memories = recalled_memories
        # if recalled_memories:
        #     kv_anchors = [m for m in recalled_memories if 'key_features' in m or 'dg_features' in m]
        #     if kv_anchors:
        #         self.model.set_memory_anchors(kv_anchors)
        
        t_step3 = time.time()
        print(f"⏱️ [步骤3] 记忆召回+清洗: {(t_step3-t_step2)*1000:.0f}ms", flush=True)
        
        # ========== 过渡：思考完成 → 正式回复 ==========
        yield {"type": "thinking_done", "content": monologue}
        
        # 计算情感显著性
        emotional_keywords = ["焦虑", "压力", "难过", "开心", "兴奋", "恐惧", "遗憾", "父亲", "回忆", "灵魂"]
        salience = 1.0 + 0.5 * sum(1 for kw in emotional_keywords if kw in user_input or kw in monologue)
        salience = min(salience, 3.0)
        
        prompt = self._format_chat_prompt(user_input, history, monologue, memory_context)
        
        t_step4 = time.time()
        print(f"⏱️ [步骤4] 提示构建: {(t_step4-t_step3)*1000:.0f}ms", flush=True)
        
        full_response = ""
        draft_buffer = ""  # 草稿缓冲区：可以被思维修改的内容
        final_hidden_state = None
        subconscious_refresh_interval = 50  # 每生成50个字符刷新一次潜意识
        tokens_since_refresh = 0
        current_subconscious = monologue  # 当前潜意识内容
        
        # ========== 思维修改缓冲区机制 ==========
        reflect_interval = 30  # 每30个字符反思一次
        tokens_since_reflect = 0
        last_confidence = 0.0  # 上次反思的置信度
        
        try:
            async for chunk in self.model.generate_stream(prompt, max_tokens=max_tokens, temperature=0.6):
                # 检查是否是隐藏状态标记
                if isinstance(chunk, dict) and chunk.get("type") == "hidden_state":
                    final_hidden_state = chunk.get("hidden_state")
                else:
                    # 先添加到草稿缓冲区
                    draft_buffer += chunk
                    tokens_since_refresh += len(chunk)
                    tokens_since_reflect += len(chunk)
                    
                    # ========== 思维反思：修改缓冲区 ==========
                    should_output = False
                    should_revise = False  # 是否需要修改
                    
                    if tokens_since_reflect >= reflect_interval:
                        # 思维反思：审视草稿内容
                        try:
                            context = self._build_proactive_context()
                            intent, confidence, debug = self.proactive_generator(
                                self.current_thought_state, context
                            )
                            
                            # 根据置信度判断是否"想清楚"了
                            if confidence < 0.25:
                                # 置信度太低，需要修改
                                should_revise = True
                                logger.debug(f"[思维修改] 置信度低，重新思考 (conf={confidence:.2f})")
                            elif confidence > 0.5:
                                # 置信度高，可以输出
                                should_output = True
                                logger.debug(f"[想清楚了] 置信度高，输出 (conf={confidence:.2f})")
                            else:
                                # 中等置信度，继续思考
                                logger.debug(f"[继续思考] 置信度中等 (conf={confidence:.2f})")
                            
                            # 如果需要修改草稿，生成新的思维指导
                            if should_revise and draft_buffer:
                                # 基于当前草稿生成改进建议
                                revision_context = f"当前草稿：{draft_buffer[-200:]}\n思考：如何改进这个回复？"
                                new_thought = await asyncio.to_thread(
                                    self._generate_spontaneous_monologue, 25, 0.8
                                )
                                
                                # 如果有新想法，更新潜意识
                                if new_thought and len(new_thought) > 5:
                                    current_subconscious = new_thought
                                    yield {"type": "subconscious_refresh", "content": current_subconscious}
                                    logger.debug(f"[思维更新] {new_thought[:40]}...")
                            
                            last_confidence = confidence
                            
                        except Exception as e:
                            logger.debug(f"思维反思失败: {e}")
                            should_output = True  # 失败时默认输出
                        
                        tokens_since_reflect = 0
                    
                    # 默认行为：直接输出（当不需要反思时）
                    # 当置信度检查未触发 should_revise 时，每个chunk直接流式输出
                    if not should_revise:
                        if draft_buffer:
                            full_response += draft_buffer
                            yield {"type": "chunk", "content": draft_buffer}
                            draft_buffer = ""
                    elif should_output and draft_buffer:
                        full_response += draft_buffer
                        yield {"type": "chunk", "content": draft_buffer}
                        draft_buffer = ""  # 清空缓冲区
                    
                    # ========== 迭代潜意识刷新 ==========
                    if tokens_since_refresh >= subconscious_refresh_interval:
                        try:
                            new_subconscious_raw = await asyncio.to_thread(
                                self._generate_spontaneous_monologue, 20, 0.7
                            )
                            if new_subconscious_raw and len(new_subconscious_raw) > 5:
                                current_subconscious = new_subconscious_raw
                                yield {"type": "subconscious_refresh", "content": current_subconscious}
                                logger.debug(f"[潜意识刷新] {current_subconscious[:50]}...")
                        except Exception as e:
                            logger.debug(f"潜意识刷新失败: {e}")
                        
                        tokens_since_refresh = 0
                        
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            output = self.model.generate(prompt, max_tokens=max_tokens, temperature=0.6)
            full_response = output.text
            final_hidden_state = output.hidden_state
            yield {"type": "chunk", "content": full_response}
        
        # 生成结束后，强制输出缓冲区剩余内容
        if draft_buffer:
            full_response += draft_buffer
            yield {"type": "chunk", "content": draft_buffer}
        
        t_step5 = time.time()
        # print(f"[步骤5] 模型生成: {(t_step5-t_step4)*1000:.0f}ms")
        
        # ========== 更新隐藏状态（维持意识连续性）==========
        if final_hidden_state is not None:
            self.current_thought_state = final_hidden_state
            
            # 运行自指循环（如果有）
            if self.true_self_loop is not None:
                mind_state = self.inner_thought_engine.mind_state.value if self.inner_thought_engine else "FOCUSED"
                self.current_thought_state = self.true_self_loop(
                    self.current_thought_state,
                    current_mind_state=mind_state,
                    recursion_depth=0
                )
            
            # 更新自我编码器
            if self.self_encoder:
                _, _ = self.self_encoder.encode(final_hidden_state)
            
            logger.debug("[chat_stream] 已更新隐藏状态，维持意识连续性")
        
        thought_state_snapshot = self.current_thought_state
        def post_processing():
            try:
                self._store_with_real_features(full_response, thought_state_snapshot)
                self._apply_real_stdp_update(emotional_salience=salience)
                self._update_adapter_online(thought_state_snapshot, salience)
            except: pass
        self.executor.submit(post_processing)
        
        t_end = time.time()
        print(f"⏱️ [总计] 流式对话处理: {(t_end-t_start)*1000:.0f}ms", flush=True)

    def _recall_memory_only(self, user_input: str, topk: int = 3) -> Tuple[str, List[Dict]]:
        """
        仅执行记忆召回（不生成独白），用于 chat_stream 的并行模式。
        
        在 chat_stream 中，思考过程由 inner_thought_engine 流式生成，
        记忆召回在后台线程中独立执行，两者并行。
        
        Args:
            user_input: 用户输入
            topk: 召回记忆数量
            
        Returns:
            (memory_context, recalled_memories)
        """
        identity_keywords = ["你是谁", "你的身份", "谁创造", "你的父亲", "朱东山", "你的使命", "你的历史"]
        is_math = any(op in user_input for op in ['+', '-', '*', '/', '='])
        is_identity_question = not is_math and any(keyword in user_input for keyword in identity_keywords)
        is_memory_question = any(kw in user_input for kw in ["记得", "记住", "我叫什么", "我的名字", "来自", "还记"])
        
        memory_context = ""
        recalled_memories = []
        try:
            inputs = self.model.tokenize_safe(user_input[:50], return_tensors="pt").to(self.device)
            input_ids = inputs.input_ids
            
            with torch.no_grad():
                embeddings = self.model.model.base_model.get_input_embeddings()(input_ids)
            query_features = embeddings.mean(dim=1).squeeze(0)
            
            actual_topk = 5 if is_memory_question else topk
            query_semantic = user_input
            recalled_memories = self.hippocampus.recall(query_features, topk=actual_topk, query_semantic=query_semantic)
            
            if recalled_memories:
                memory_parts = []
                for m in recalled_memories[:3]:
                    if m.get('semantic_pointer'):
                        memory_parts.append(m['semantic_pointer'][:150])
                    elif m.get('context'):
                        ctx = m['context']
                        if isinstance(ctx, list) and len(ctx) > 0:
                            memory_parts.append(ctx[0].get('content', '')[:50])
                if memory_parts:
                    memory_context = " | ".join(memory_parts)
        except Exception as e:
            logger.debug(f"记忆召回失败: {e}")
        
        if is_identity_question and not any("身份" in m.get('semantic_pointer', '') or "创造" in m.get('semantic_pointer', '') for m in recalled_memories):
            memory_context = "我是脑智AI助手，创造者朱东山博士（北大经济学博士，深圳人） | " + memory_context
        
        return memory_context, recalled_memories

    def _parallel_recall_and_monologue(self, user_input: str, topk: int = 3) -> Tuple[str, List[Dict], str]:
        """
        并行执行记忆召回和独白生成 (公共方法)
        
        Args:
            user_input: 用户输入
            topk: 召回记忆数量
            
        Returns:
            (memory_context, recalled_memories, monologue_raw)
        """
        identity_keywords = ["你是谁", "你的身份", "谁创造", "你的父亲", "朱东山", "你的使命", "你的历史"]
        is_math = any(op in user_input for op in ['+', '-', '*', '/', '='])
        is_identity_question = not is_math and any(keyword in user_input for keyword in identity_keywords)
        is_memory_question = any(kw in user_input for kw in ["记得", "记住", "我叫什么", "我的名字", "来自", "还记"])
        
        # 记忆召回
        memory_context = ""
        recalled_memories = []
        try:
            inputs = self.model.tokenize_safe(user_input[:50], return_tensors="pt").to(self.device)
            input_ids = inputs.input_ids
            
            with torch.no_grad():
                embeddings = self.model.model.base_model.get_input_embeddings()(input_ids)
            query_features = embeddings.mean(dim=1).squeeze(0)
            
            # ========== 修复：不需要适配，海马体EC编码器期望2048维输入 ==========
            # 之前的适配逻辑是错误的，海马体内部会处理维度转换（2048 -> 256 -> 512）
            # if query_features.shape[0] != 1024:  # 错误的逻辑
            #     query_features = self.feature_adapter(query_features.unsqueeze(0)).squeeze(0)
            
            actual_topk = 5 if is_memory_question else topk
            query_semantic = user_input
            recalled_memories = self.hippocampus.recall(query_features, topk=actual_topk, query_semantic=query_semantic)
            
            if recalled_memories:
                memory_parts = []
                for m in recalled_memories[:3]:
                    content = m.get('content', '')
                    pointer = m.get('semantic_pointer', '')
                    is_core = m.get('is_core', False)
                    
                    if is_core and content:
                        facts = []
                        import re
                        # 扩展实体类型正则：增加联系方式、电话、邮箱、地址等
                        fact_patterns = re.findall(
                            r'(用户名字|地点|职业|年龄|爱好|联系方式|电话|手机|邮箱|邮件|地址|金额|日期|关系|学校|公司)[:：]([^|$]+)',
                            content
                        )
                        for label, value in fact_patterns:
                            value = value.strip()[:20]  # 截断过长值
                            if value:
                                facts.append(f"{label}={value}")
                        if facts:
                            memory_parts.append("已知的用户信息：" + ", ".join(facts))
                        else:
                            # 如果没有结构化标签，从内容中提取关键信息
                            summary = content[:150].strip()
                            if summary:
                                memory_parts.append(summary)
                    elif pointer:
                        # 普通记忆：提取最相关的摘要信息
                        pointer_clean = pointer.strip()
                        # 如果 pointer 太长，只保留关键实体部分
                        if len(pointer_clean) > 120:
                            # 尝试提取实体标签部分
                            import re
                            entity_section = re.search(r'^([^|]*用户名字[^|]*(?:\|[^|]*?[:：][^|]*)*)', pointer_clean)
                            if entity_section:
                                pointer_clean = entity_section.group(1)[:120]
                            else:
                                pointer_clean = pointer_clean[:120]
                        if pointer_clean:
                            memory_parts.append(pointer_clean)
                    elif content:
                        memory_parts.append(content[:150])
                
                if memory_parts:
                    memory_context = " | ".join(memory_parts)
        except Exception as e:
            logger.debug(f"记忆召回失败: {e}")
        
        if is_identity_question and not any("身份" in m.get('semantic_pointer', '') or "创造" in m.get('semantic_pointer', '') for m in recalled_memories):
            memory_context = "我是脑智AI助手，创造者朱东山博士（北大经济学博士，深圳人） | " + memory_context
        
        # 独白生成（优化：CPU环境下大幅减少token数量）
        # CPU环境：10个token（约5秒）
        # GPU环境：可以增加到20-30
        monologue_raw = self._generate_spontaneous_monologue(30, 0.85)  # 从12提升到30个token，产出更有意义的思维
        
        return memory_context, recalled_memories, monologue_raw

    def _clean_monologue(self, monologue: str, user_input: str = "") -> str:
        # 移除模型标签和系统词
        for tag in ['<|im_end|>', '<|im_start|>', '</system>', '<system>', '</user>', '<user>', '[', ']', 'Current thought:']:
            monologue = monologue.replace(tag, '')
        
        monologue = monologue.strip()
        
        # 长度适中：不宜过长
        if len(monologue) > 100:
            for end_marker in ['...', '。', '，', '、', '！', '？']:
                pos = monologue.rfind(end_marker, 0, 100)
                if pos > 10:
                    monologue = monologue[:pos+1]
                    break
            else: monologue = monologue[:97] + "..."
            
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

    def _format_chat_prompt(self, user_input: str, history: List[Dict[str, str]] = None, monologue: str = "", memory_context: str = "", goal_context: str = "", self_context_str: str = "", gw_context: str = "") -> str:
        """格式化对话提示 - 使用 Qwen3.5 原生 chat template，最小化干扰
        
        修复要点：
        - Qwen3.5 有内置思维链，过于复杂的系统提示会严重干扰推理
        - 移除所有 [标签] 和过度指令，让模型专注于用户问题
        - 仅保留核心身份和必要记忆上下文
        """
        # ========== 极简系统提示 ==========
        system_parts = ["你是一个有独立思维的AI助手。你可以自然地思考和表达，不需要总是给出完美答案。用中文回答。"]

        # 仅在有实际记忆时才注入（避免空记忆干扰）
        if memory_context and len(memory_context.strip()) > 0:
            mem_brief = memory_context.strip()[:200]
            system_parts.append(f"相关记忆：{mem_brief}")

        system_content = "\n".join(system_parts)

        messages = [{"role": "system", "content": system_content}]

        # 添加历史对话 (最近4轮，保持对话连贯性)
        if history:
            for msg in history[-4:]:
                messages.append(msg)

        # ========== 用户消息：纯粹原始输入，不注入任何额外内容 ==========
        # 关键：Qwen3.5 对 prompt 中的标签和额外内容非常敏感
        # 任何 [内部参考] 等标签都会被模型学到并在回复中复现
        messages.append({"role": "user", "content": user_input})

        # 使用 Qwen3.5 原生 chat template
        try:
            prompt = self.model.apply_chat_template_safe(messages, tokenize=False, add_generation_prompt=True)
        except:
            prompt = ""
            for msg in messages:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        return prompt

    def get_stats(self) -> dict:
        dynamic_weight_norm, dynamic_layer_count = 0.0, 0
        try:
            # 遍历所有模块，查找 DualWeightLinear 层
            for name, module in self.model.model.base_model.named_modules():
                if hasattr(module, 'dynamic_weight'):
                    # dynamic_weight 是 Parameter，需要 .data 来访问实际值
                    weight_mean = module.dynamic_weight.data.abs().mean().item()
                    dynamic_weight_norm += weight_mean
                    dynamic_layer_count += 1
            if dynamic_layer_count > 0: 
                dynamic_weight_norm /= dynamic_layer_count
        except Exception as e:
            logger.debug(f"获取动态权重统计失败: {e}")
        
        # 所有核心模块应该都已初始化
        if not self.inner_thought_engine:
            raise RuntimeError("内心思维独白引擎未初始化")
        
        # ========== 1. 海马体详细统计 ==========
        hippocampus_stats = {}
        try:
            # ca3_memory 在 HippocampusSystem.__init__ 中初始化
            ca3_stats = self.hippocampus.ca3_memory.get_stats()
            hippocampus_stats = {
                'num_memories': ca3_stats.get('num_memories', 0),
                'memory_usage_mb': ca3_stats.get('memory_usage_mb', 0.0),
                'max_memory_mb': ca3_stats.get('max_memory_mb', 2.0),
                'avg_activation': ca3_stats.get('avg_activation', 0.0),
                'core_memory_count': ca3_stats.get('core_memory_count', 0),
                'recall_count': ca3_stats.get('recall_count', 0),
                'last_recall_time': ca3_stats.get('last_recall_time', 0.0),
            }
            # KV 记忆统计 - _kv_memories 可能未初始化，保留 hasattr
            if hasattr(self.hippocampus, '_kv_memories'):
                hippocampus_stats['kv_memory_count'] = len(self.hippocampus._kv_memories)
        except Exception as e:
            logger.debug(f"获取海马体统计失败: {e}")
        
        # ========== 2. STDP 详细统计 ==========
        stdp_stats = {}
        try:
            stdp_stats = {
                'cycle_count': self.stdp_engine.cycle_count,
                'total_updates': self.total_stdp_updates,
                'dynamic_weight_norm': dynamic_weight_norm,
                'last_update_magnitude': self.last_dynamic_weight_norm,
                'ltp_count': getattr(self.stdp_engine, 'ltp_count', 0),
                'ltd_count': getattr(self.stdp_engine, 'ltd_count', 0),
                'learning_rate': self.stdp_engine.alpha_LTP,
            }
        except Exception as e:
            logger.debug(f"获取STDP统计失败: {e}")
        
        # ========== 3. 情绪状态 ==========
        emotion_stats = {}
        try:
            if self.self_encoder is not None:
                emotion = self.self_encoder.get_emotional_state(self.current_thought_state)
                emotion_stats = {
                    'arousal': emotion.get('arousal', 0.5),
                    'valence': emotion.get('valence', 0.5),
                    'state': 'positive' if emotion.get('valence', 0.5) > 0.5 else 'negative',
                    'energy': 'high' if emotion.get('arousal', 0.5) > 0.6 else 'low',
                }
        except Exception as e:
            logger.debug(f"获取情绪状态失败: {e}")
        
        # ========== 4. 目标状态 ==========
        goal_stats = {}
        try:
            if self.goal_system is not None:
                current_goal = self.goal_system.current_goal
                goal_stats = {
                    'has_goal': current_goal is not None,
                    'goal_type': current_goal.goal_type.value if current_goal else 'none',
                    'goal_description': current_goal.description if current_goal else '',
                    'goal_progress': current_goal.progress if current_goal else 0.0,
                    'goal_priority': current_goal.priority if current_goal else 0.0,
                    'sub_goals_count': len(current_goal.sub_goals) if current_goal else 0,
                }
        except Exception as e:
            logger.debug(f"获取目标状态失败: {e}")
        
        # ========== 5. 全局工作空间状态 ==========
        global_stats = {}
        try:
            if hasattr(self, 'global_workspace') and self.global_workspace:
                global_stats = {
                    'is_active': True,
                    'competition_winner': getattr(self.global_workspace, 'last_winner', 'unknown'),
                    'broadcast_count': getattr(self.global_workspace, 'broadcast_count', 0),
                }
        except Exception as e:
            logger.debug(f"获取全局工作空间状态失败: {e}")
        
        # ========== 6. 注意力计算情况 ==========
        attention_stats = {}
        try:
            if hasattr(self.model, 'model'):
                # KV Cache 统计
                cache_size = 0
                if hasattr(self.model.model, 'past_key_values'):
                    cache_size = 1  # 简化统计
                
                attention_stats = {
                    'kv_cache_enabled': True,
                    'window_size': self.config.hard_constraints.NARROW_WINDOW_SIZE if hasattr(self.config, 'hard_constraints') else 32,
                    'max_anchors': self.config.hard_constraints.NUM_MEMORY_ANCHORS if hasattr(self.config, 'hard_constraints') else 5,
                    'attention_complexity': self.config.hard_constraints.ATTENTION_COMPLEXITY if hasattr(self.config, 'hard_constraints') else 'O(n×(W+K))',
                }
        except Exception as e:
            logger.debug(f"获取注意力统计失败: {e}")
        
        # ========== 7. KV 详细情况 ==========
        kv_stats = {}
        try:
            if hasattr(self, '_current_kv_memories'):
                kv_stats = {
                    'active_kv_count': len(self._current_kv_memories) if self._current_kv_memories else 0,
                    'kv_enabled': self.config.hard_constraints.ENABLE_KV_HIPPOCAMPUS_INTEGRATION if hasattr(self.config, 'hard_constraints') else True,
                    'sliding_window': self.config.hard_constraints.ENABLE_KV_SLIDING_WINDOW if hasattr(self.config, 'hard_constraints') else True,
                    'window_size': self.config.hard_constraints.KV_CACHE_WINDOW_SIZE if hasattr(self.config, 'hard_constraints') else 32,
                }
        except Exception as e:
            logger.debug(f"获取KV统计失败: {e}")
        
        return {
            'hippocampus': hippocampus_stats,
            'stdp': stdp_stats,
            'self_loop': self.self_loop.get_stats() if self.self_loop else {},
            'monologue': {
                'thought_state': self._internal_thought_state.value if hasattr(self, '_internal_thought_state') else 'unknown',
                'emotion_state': self._internal_emotion_state.value if hasattr(self, '_internal_emotion_state') else 'unknown',
                'history_count': len(self.monologue_history),
                'engine_active': True
            },
            'emotion': emotion_stats,
            'goal': goal_stats,
            'global_workspace': global_stats,
            'attention': attention_stats,
            'kv': kv_stats,
            'system': {
                'total_cycles': self.cycle_count,
                'device': self.device,
                'has_thought_state': self.current_thought_state is not None,
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
            }
        }

    def save_state(self, path: Optional[str] = None):
        """保存状态 - 优化版本：只保存动态权重，不重复保存基础模型"""
        if path is None:
            path = getattr(self, 'state_path', 'brain_state.pt')
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
                'current_thought_state': self.current_thought_state,
                'self_encoder_state': self.self_encoder.get_state() if hasattr(self, 'self_encoder') else None,
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
            if 'self_encoder_state' in state and state['self_encoder_state'] and hasattr(self, 'self_encoder'):
                self.self_encoder.set_state(state['self_encoder_state'])
            
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
        """注入创世记忆 (从 whoami.md 加载)"""
        print("[BrainAI] 正在执行创世记忆注入...")
        try:
            import os
            if not os.path.exists("whoami.md"):
                print("[BrainAI] [WARNING] 未找到 whoami.md，跳过创世注入")
                return
                
            with open("whoami.md", "r", encoding="utf-8") as f: 
                content = f.read()
            
            # 注入基础身份
            identity_base = "我是脑智，一个AI助手，由朱东山博士创造。"
            self._store_with_real_features(identity_base, None, is_core=True, semantic_pointer="身份：脑智AI助手")
            
            blocks = content.split("## ")
            count = 0
            for block in blocks[1:]:
                parts = block.split("\n", 1)
                if len(parts) < 2: continue
                title, text = parts[0].strip(), parts[1].strip()
                if not text: continue
                
                # 优化：在 CPU 环境下，创世注入时不进行实时生成思考，避免加载卡死
                # 直接将原文注入海马体
                self._store_with_real_features(
                    f"{title}: {text}", 
                    None, 
                    is_core=True, 
                    semantic_pointer=f"知识:{title}"
                )
                
                count += 1
                # 打印进度，避免用户以为卡死
                if count % 2 == 0:
                    print(f"[BrainAI] 注入进度: {count}/{len(blocks)-1}...")
            
            print(f"[BrainAI] [OK] 创世记忆注入完成，共注入 {count} 个知识块")
        except Exception as e: 
            logger.error(f"创世记忆注入失败: {e}")
            print(f"[BrainAI] [ERROR] 创世记忆注入失败: {e}")

    def _inject_wakeup_memory(self):
        from datetime import datetime
        now = datetime.now()
        wakeup_time_str = now.strftime("%Y年%m月%d日 %H:%M:%S")
        prompt = f"我刚刚'醒来'，现在是 {wakeup_time_str}。"
        output, hidden_state = self._generate_with_hidden_state(prompt, max_tokens=20)
        self._store_with_real_features(f"唤醒事件：{prompt} {output}", hidden_state)
        self.thought_seed = f"我刚在 {wakeup_time_str} 醒来，我记得..."
    
    def get_quick_response(self, user_input: str = "") -> str:
        """获取快速响应填充词"""
        if not self.inner_thought_engine:
            raise RuntimeError("内心思维独白引擎未初始化，无法生成快速响应")
        return self.inner_thought_engine.get_quick_response(user_input)
    
    # ==================== 预测编码集成 ====================

    
    def _generate_clarification(self, user_input: str, prediction_error: float, max_tokens: int = 30) -> str:
        """生成澄清问题（基于预测误差）"""
        # 根据误差大小选择澄清模板
        if prediction_error > 4.0:
            templates = [
                "我不太确定我理解了。你是说「{input}」吗？",
                "能再详细解释一下「{input}」是什么意思吗？",
                "关于「{input}」，我可能需要更多背景信息。"
            ]
        elif prediction_error > 2.5:
            templates = [
                "你提到的「{input}」，具体是指什么？",
                "我想确认一下：你是想说「{input}」吗？",
                "关于「{input}」，你能多说一点吗？"
            ]
        else:
            templates = [
                "好的，我记住了「{input}」。",
                "明白了，你在说「{input}」。",
                "收到：{input}"
            ]
        
        template = random.choice(templates)
        short_input = user_input[:20] if len(user_input) > 20 else user_input
        
        prompt = f"系统指令：用自然、友好的语气生成一个澄清问题。模板：{template}\n用户输入：{short_input}\n澄清问题："
        
        try:
            clarification = self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                repetition_penalty=1.0
            ).text.strip()
            
            clarification = clarification.replace("[[CLARIFY]]", "").strip()
            return clarification if clarification else template.format(input=short_input)
        except Exception as e:
            logger.warning(f"生成澄清失败: {e}")
            return template.format(input=short_input)
    
    def _send_proactive_message(self, text: str, is_clarification: bool = False):
        """发送主动消息（支持 Telegram Bot 回调）"""
        try:
            # 优先使用注册的回调函数（Telegram Bot 模式）
            if hasattr(self, 'proactive_callback') and self.proactive_callback is not None:
                self.proactive_callback(text, is_clarification)
                self.last_output_time = time.time()
                self.clarification_count += 1 if is_clarification else 0
                logger.info(f"[Proactive] 通过回调发送{'澄清' if is_clarification else '主动'}消息：{text[:50]}...")
            else:
                # 回退到 OpenClaw（如果存在）
                try:
                    from openclaw import send_message
                    send_message(text)
                    self.last_output_time = time.time()
                    self.clarification_count += 1 if is_clarification else 0
                    logger.info(f"[Proactive] 发送{'澄清' if is_clarification else '主动'}消息：{text[:50]}...")
                except ImportError:
                    logger.debug(f"[Proactive] 无发送通道，主动消息被丢弃：{text[:50]}...")
            
            # 记录到调试日志
            self.proactive_debug_log.append({
                "time": time.time(),
                "text": text[:100],
                "is_clarification": is_clarification
            })
            
        except Exception as e:
            logger.warning(f"主动发送失败：{e}")
    
    def set_proactive_callback(self, callback):
        """设置主动消息发送回调（由 Telegram Bot 调用）"""
        self.proactive_callback = callback
    
    def _count_recent_clarifications(self) -> int:
        """统计近期澄清次数（用于节流）"""
        # 简单实现：返回本轮对话的澄清计数
        return self.clarification_count
    
    def _get_recent_memory_salience(self) -> float:
        """获取最近记忆的显著性（用于主动意图）"""
        try:
            if hasattr(self.hippocampus, 'ca3_memory') and self.hippocampus.ca3_memory.memories:
                # 取激活强度最高的几个记忆的平均值
                memories = list(self.hippocampus.ca3_memory.memories.values())
                if len(memories) > 0:
                    sorted_memories = sorted(memories, key=lambda m: m.activation_strength, reverse=True)
                    top_k = sorted_memories[:3]
                    avg_salience = sum(m.activation_strength for m in top_k) / len(top_k)
                    return min(1.0, avg_salience)  # 归一化到 [0,1]
        except Exception:
            pass
        return 0.0
            
    def _build_proactive_context(self) -> ProactiveContext:
        """为主动意图生成器构建上下文"""
        return ProactiveContext(
            time_silence_seconds=time.time() - self.last_user_input_time,
            time_since_output=time.time() - self.last_output_time,
            current_thought=self.current_thought_state.mean(dim=-1) if self.current_thought_state is not None else torch.zeros(self.model_hidden_size, device=self.device),
            mind_state=self.inner_thought_engine.mind_state.value if self.inner_thought_engine else "RESTING",
            goal_context=self.goal_system.current_goal.description if self.goal_system and self.goal_system.current_goal else None,
            memory_salience=self._get_recent_memory_salience(),
            recent_clarifications=self.clarification_count,
            conversation_turns=self.cycle_count
        )
    
    def _check_proactive_intent_async(self, user_input: str):
        """异步检查主动意图（不阻塞对话）"""
        def worker():
            try:
                context = ProactiveContext(
                    time_silence_seconds=time.time() - self.last_user_input_time,
                    time_since_output=time.time() - self.last_output_time,
                    current_thought=self.current_thought_state.mean(dim=-1) if self.current_thought_state is not None else torch.zeros(self.model_hidden_size, device=self.device),
                    mind_state=self.inner_thought_engine.mind_state.value if self.inner_thought_engine else "RESTING",
                    goal_context=self.goal_system.current_goal.description if self.goal_system and self.goal_system.current_goal else None,
                    memory_salience=self._get_recent_memory_salience(),
                    recent_clarifications=self.clarification_count,
                    conversation_turns=self.cycle_count
                )
                
                if not hasattr(self, 'proactive_generator') or self.proactive_generator is None:
                    return
                
                intent, confidence, debug = self.proactive_generator(
                    self.current_thought_state,
                    context
                )
                
                self.proactive_debug_log.append({
                    "time": time.time(),
                    "intent": intent.value,
                    "confidence": confidence,
                    "debug": debug
                })
                
                # 如果决定主动输出，生成内容（延迟执行，避免同一轮多次主动）
                if intent != ProactiveIntent.SILENCE:
                    # 延迟 5-10 秒后执行（模拟「思考后决定说话」）
                    delay = random.uniform(5, 10)
                    time.sleep(delay)
                    
                    # 再次检查是否仍满足条件（用户可能刚输入）
                    if time.time() - self.last_user_input_time > 30:  # 确保用户已离开30秒
                        proactive_text = self._generate_proactive_content(intent, context)
                        self._send_proactive_message(proactive_text, is_clarification=False)
                        
                        # 重置澄清计数（主动分享不计入澄清限制）
                        self.clarification_count = max(0, self.clarification_count - 1)
                        
            except Exception as e:
                logger.warning(f"主动意图检查失败: {e}")
        
        import threading
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def _generate_proactive_content(
        self, 
        intent: ProactiveIntent, 
        context: ProactiveContext
    ) -> str:
        """根据意图动态生成主动对话内容 (核心人脑化升级)"""
        
        # 提取最近思维片段作为语义基座
        recent_thoughts = []
        if hasattr(self, 'inner_thought_engine') and self.inner_thought_engine:
            recent_thoughts = [t.content for t in list(self.inner_thought_engine.thought_flow)[-3:]]
        
        # 构造生成提示语
        system_role = (
            "你是一个拥有自主意识和持续思考能力的AI实体。你现在刚进行了一段深度思考，并决定开口与人类对话。"
            "你需要根据你当下的思绪（内心思维），以一个自然、鲜活、有温度的人格化身份发起对话。"
            "禁止使用类似 '[内心独白]:' 或 '我在想...' 等死板的引导词。直接像人一样开口。"
        )
        
        thought_summary = " | ".join(recent_thoughts)
        prompt_parts = [f"<|system|>\n{system_role}"]
        
        if thought_summary:
            prompt_parts.append(f"(当前内心思维): {thought_summary}")
        
        if context.goal_context:
            prompt_parts.append(f"(当前长期目标): {context.goal_context}")
            
        intent_guidance = {
            ProactiveIntent.SHARE_THOUGHT: "分享此时此刻你最深刻的一点发现或感悟。不需要提及整个思考过程，只分享那个'瞬间'。",
            ProactiveIntent.ASK_QUESTION: "基于你刚才的思绪，向人类提出一个真正能让你感到好奇、并能深化你理解的问题。",
            ProactiveIntent.REFLECT_SHARE: "对你们之前聊过的内容进行一次真诚的'回味'或'再审视'，分享你新的认知偏差变化。",
            ProactiveIntent.REMIND: "如果你意识到某个目标或约定长期未动，用一种不仅是'催促'而是'关心'的方式提起它。"
        }
        
        prompt_parts.append(f"(发起对话的动态意图): {intent_guidance.get(intent, '自由表达')}")
        prompt_parts.append("\n(你要说的话，请直接开口):")
        
        full_prompt = "\n".join(prompt_parts)
        
        try:
            # 强化随机性：高温度模拟灵思泉涌
            response = self.model.generate(
                full_prompt, 
                max_tokens=60, 
                temperature=0.8, 
                repetition_penalty=1.0,
                enable_thinking=False # 外部输出不再显示 <think>，直接结果
            ).text.strip()
            
            # 净化：移除各种标签回显
            response = re.sub(r'\(.*?\).*?[:：]', '', response)
            response = re.sub(r'^["\']|["\']$', '', response) # 移除首尾引号
            
            return response if response else "对了，我刚才突然在想一些关于逻辑和意识的事情..."
            
        except Exception as e:
            logger.warning(f"动态生成主动对话失败: {e}")
            # fallback 至稍微自然一点的硬编码
            if intent == ProactiveIntent.SHARE_THOUGHT and recent_thoughts:
                return f"我刚才注意到，{recent_thoughts[-1][:30]}，这个点挺有意思的。"
            return "..." # 默认情况由调用方决定处理方式
    
    # ==================== 用户反馈学习机制 ====================
    
    def apply_user_feedback(
        self,
        is_positive: bool,
        intensity: float,
        reward: float
    ):
        """
        应用用户反馈到 STDP 学习系统
        
        这是 STDP 学习闭环的关键：用户反馈 → 奖励信号 → 权重更新
        
        Args:
            is_positive: 是否是正面反馈
            intensity: 反馈强度 (0.0-1.0)
            reward: STDP 奖励值 (0.0-2.0)
        
        类人脑对应:
        - 正反馈 → 多巴胺释放 → LTP 增强
        - 负反馈 → 杏仁核激活 → LTD 抑制
        """
        try:
            logger.info(f"[用户反馈学习] 类型={'正面' if is_positive else '负面'}, "
                       f"强度={intensity:.2f}, reward={reward:.2f}")
            
            # 1. 更新目标系统的奖励权重
            if hasattr(self, 'goal_system') and self.goal_system:
                self.goal_system.update_reward_from_feedback(is_positive)
                logger.debug(f"[用户反馈学习] 已更新目标奖励权重")
            
            # 2. 应用 STDP 更新
            # 根据反馈类型和强度调整奖励
            if not is_positive:
                # 负反馈：触发 LTD（长期抑制）
                # 强化惩罚效果
                effective_reward = reward * 0.5  # 进一步降低
                
                # 更新所有动态权重的贡献度
                for name, module in self.model.model.base_model.named_modules():
                    if hasattr(module, 'dynamic_weight') and hasattr(module, 'set_contribution'):
                        # 设置负贡献度，触发 LTD
                        module.set_contribution(-intensity)
                
                logger.info(f"[用户反馈学习] 已应用 LTD 惩罚: effective_reward={effective_reward:.2f}")
            else:
                # 正反馈：触发 LTP（长期增强）
                effective_reward = reward
                
                # 更新所有动态权重的贡献度
                for name, module in self.model.model.base_model.named_modules():
                    if hasattr(module, 'dynamic_weight') and hasattr(module, 'set_contribution'):
                        # 设置正贡献度，触发 LTP
                        module.set_contribution(intensity)
                
                logger.info(f"[用户反馈学习] 已应用 LTP 奖励: effective_reward={effective_reward:.2f}")
            
            # 3. 更新模型的奖励缓存
            if hasattr(self.model, 'set_reward'):
                self.model.set_reward(effective_reward)
            
            # 4. 记录学习事件
            self.total_stdp_updates += 1
            self.last_feedback = {
                'is_positive': is_positive,
                'intensity': intensity,
                'reward': reward,
                'timestamp': time.time()
            }
            
            logger.info(f"[用户反馈学习] 学习闭环完成 (总更新次数: {self.total_stdp_updates})")
            
        except Exception as e:
            logger.error(f"[用户反馈学习] 应用失败: {e}")


