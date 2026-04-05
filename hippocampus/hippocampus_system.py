"""
海马体记忆系统 - 完整集成

将 EC、DG、CA3、CA1、SWR 五个子模块集成为完整的类脑海马体系统
提供统一的编码、召回、门控接口
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import time
import logging

logger = logging.getLogger(__name__)

from .ec_encoder import EntorhinalEncoder
from .dg_separator import DentateGyrusSeparator
from .ca3_memory import CA3EpisodicMemory
from .ca1_gate import CA1AttentionGate
from .swr_consolidation import SWRConsolidation
from .semantic_engine import SemanticSummarizer
from .memory_layers import TierConfig


class HippocampusSystem(nn.Module):
    """
    完整的类人脑海马体系统
    
    整合五个核心子模块:
    - EC (内嗅皮层): 特征编码
    - DG (齿状回): 模式分离
    - CA3: 情景记忆存储 + 模式补全
    - CA1: 时序编码 + 注意力门控
    - SWR: 离线回放巩固
    
    输入：模型注意力层输出的 token 特征
    输出：1-2 个记忆锚点 + 注意力门控信号
    """
    
    def __init__(self, config, device: Optional[str] = None):
        super().__init__()
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从配置加载参数
        hc_config = config.hippocampus
        
        # ========== 1. EC 内嗅皮层 - 特征编码 ==========
        model_hidden_size = getattr(config, 'model_hidden_size', 2048)
        self.ec_encoder = EntorhinalEncoder(
            input_dim=model_hidden_size,
            output_dim=hc_config.EC_feature_dim,
            sparsity=hc_config.DG_sparsity,
            freeze_encoder=False
        )
        
        # ========== 2. DG 齿状回 - 模式分离 ==========
        self.dg_separator = DentateGyrusSeparator(
            input_dim=hc_config.EC_feature_dim,
            output_dim=hc_config.EC_feature_dim * 2,  # 扩展到 128 维
            sparsity=hc_config.DG_sparsity,
            orthogonalization=hc_config.DG_orthogonalization
        )
        
        # ========== 3. 语义引擎（人类记忆增强）==========
        self.semantic_engine = SemanticSummarizer(model_interface=None, device=self.device)
        
        # ========== 3.5 CA3 情景记忆库（集成语义引擎和分层）==========
        tier_config = TierConfig(
            short_term_max_capacity=hc_config.CA3_max_capacity // 5,
            mid_term_max_capacity=hc_config.CA3_max_capacity // 2,
            long_term_max_capacity=hc_config.CA3_max_capacity,
        )
        self.ca3_memory = CA3EpisodicMemory(
            max_capacity=hc_config.CA3_max_capacity,
            feature_dim=hc_config.EC_feature_dim * 2,
            timestamp_precision_ms=hc_config.CA3_timestamp_precision_ms,
            # BUG FIX: 使用统一阈值，而非硬编码0.7
            # 之前recall_threshold=0.7远高于实际embedding相似度，导致记忆在CA3层就被丢弃
            recall_threshold=hc_config.recall_threshold,
            decay_rate=0.999,
            semantic_engine=self.semantic_engine,
            tier_config=tier_config,
        )
        
        # ========== 4. CA1 注意力 gate ==========
        self.ca1_gate = CA1AttentionGate(
            feature_dim=hc_config.EC_feature_dim * 2,
            hidden_size=model_hidden_size,
            recall_topk=hc_config.recall_topk,
            temporal_encoding=hc_config.CA1_temporal_encoding,
            gate_type="additive"
        )
        
        # ========== 5. SWR 离线回放巩固 ==========
        self.swr_consolidation = SWRConsolidation(
            config=config,
            hippocampus_module=self,
            idle_threshold_s=hc_config.SWR_idle_threshold_s,
            replay_frequency=hc_config.SWR_replay_frequency
        )
        
        # ========== 6. 联想记忆网络（延迟注入，由 HumanCognitiveIntegration 初始化）==========
        self.associative_network = None  # AssociativeMemoryNetwork

        # ========== 7. 记忆重构引擎（延迟注入）==========
        self.reconstruction_engine = None  # MemoryReconstructionEngine

        # ========== 8. 梦境巩固系统（延迟注入）==========
        self.dream_system = None  # DreamConsolidationSystem

        # ========== 内存监控 ==========
        self.max_memory_bytes = hc_config.max_memory_bytes
        self.memory_usage_bytes = 0
        
        # ========== 性能优化: 缓存 ==========
        self._query_encoding_cache = {}  # 查询特征编码缓存
        self._memory_size_cache = 0  # 增量内存统计
        
        # ========== 周期计数器 ==========
        self.cycle_count = 0
        
        # ========== 内存使用追踪（预初始化，避免 hasattr 检查）==========
        self._last_memory_count = 0
    
    def encode(
        self,
        features: torch.Tensor,
        token_id: int,
        timestamp: int,
        context: Optional[List[dict]] = None,
        kv_features: Optional[Dict[str, torch.Tensor]] = None  # 新增: 用于窄带宽注意力的 KV 特征
    ) -> str:
        """
        记忆编码流程
        
        完整流程：Token 特征 → EC 编码 → DG 分离 → CA3 存储
        
        Args:
            features: Token 特征 [hidden_size] 或 [1, hidden_size]
            token_id: Token ID
            timestamp: 时间戳 (ms)
            context: 上下文信息
            kv_features: KV 特征字典 {'key': tensor, 'value': tensor}，用于窄带宽注意力
        
        Returns:
            memory_id: 生成的唯一记忆 ID
        """
        # ========== 1. 特征预处理 ==========
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # ========== 2. EC 编码 ==========
        ec_code = self.ec_encoder.encode_single(features.squeeze(0))
        
        # ========== 3. DG 模式分离 ==========
        dg_output, memory_id = self.dg_separator.separate_and_id(ec_code)
        
        # ========== 4. 构建时序骨架 ==========
        temporal_skeleton = ""
        causal_links = []
        is_core = False
        content = ""
        
        if context and len(context) > 0:
            # 从前序上下文提取时序关系
            prev_tokens = [ctx.get('token_id', -1) for ctx in context]
            temporal_skeleton = f"{prev_tokens} -> {token_id}"
            
            # 简单因果推断 (实际应更复杂)
            causal_links.append(f"preceded_by_{prev_tokens[-1]}" if prev_tokens else "start")
            
            # 提取 is_core 和 content
            for ctx in context:
                if 'is_core' in ctx:
                    is_core = ctx['is_core']
                if 'content' in ctx:
                    content = ctx['content']
        
        # ========== 5. CA3 存储 ==========
        # 优先使用context中的semantic_pointer，否则使用token_id
        semantic_pointer = f"token_{token_id}"
        if context and len(context) > 0:
            # 从context中提取semantic_pointer
            for ctx in context:
                if 'semantic_pointer' in ctx and ctx['semantic_pointer']:
                    semantic_pointer = ctx['semantic_pointer']
                    break
        
        # ========== 5.5 提取 KV 特征（如果提供）==========
        key_features = None
        value_features = None
        if kv_features:
            key_features = kv_features.get('key', None)
            value_features = kv_features.get('value', None)
            if key_features is not None:
                key_features = key_features.detach().cpu()
            if value_features is not None:
                value_features = value_features.detach().cpu()
        
        # ========== 5.6 从 context 提取 user_input / ai_response（供语义引擎生成摘要）==========
        user_input = ""
        ai_response = ""
        if context and len(context) > 0:
            for ctx in context:
                if 'user_input' in ctx and ctx['user_input']:
                    user_input = str(ctx['user_input'])
                if 'ai_response' in ctx and ctx['ai_response']:
                    ai_response = str(ctx['ai_response'])

        self.ca3_memory.store(
            memory_id=memory_id,
            timestamp=timestamp,
            semantic_pointer=semantic_pointer,
            temporal_skeleton=temporal_skeleton,
            causal_links=causal_links,
            dg_features=dg_output.detach().cpu(),
            is_core=is_core,
            content=content,
            key_features=key_features,    # 新增
            value_features=value_features,  # 新增
            user_input=user_input,        # 供语义引擎生成摘要
            ai_response=ai_response,      # 供语义引擎生成摘要
        )
        
        # ========== 6. 联想记忆：检测并创建关联 ==========
        # 在新记忆存入CA3后，自动检测与已有记忆的关联
        if self.associative_network is not None:
            try:
                new_memory = self.ca3_memory.memories.get(memory_id)
                if new_memory is not None:
                    self.associative_network.detect_and_create_associations(
                        new_memory, self.ca3_memory.memories
                    )
            except Exception as e:
                logger.debug(f"[Hippocampus] 联想关联创建失败: {e}")

        # ========== 7. 更新内存使用 ==========
        self._update_memory_usage()
        
        self.cycle_count += 1
        
        return memory_id
    
    def recall(
        self,
        query_features: torch.Tensor,
        topk: int = 2,
        query_semantic: Optional[str] = None,
        query_timestamp: Optional[int] = None
    ) -> List[dict]:
        """
        记忆召回流程（性能优化版）
        
        优化点:
        - 缓存查询特征编码结果
        - 减少不必要的拷贝和转换
        """
        # ========== 1. 特征预处理 ==========
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        
        # ========== 优化: 检查编码缓存 ==========
        # 使用特征哈希作为缓存键 - 统一转为CPU float32连续字节，避免BFloat16不兼容
        query_for_hash = query_features.detach().to('cpu').float().contiguous()
        cache_key = hash(query_for_hash.numpy().tobytes())
        
        if cache_key in self._query_encoding_cache:
            ec_code, dg_features = self._query_encoding_cache[cache_key]
        else:
            # EC 编码
            ec_code = self.ec_encoder.encode_single(query_features.squeeze(0))
            # DG 分离
            dg_features = self.dg_separator.forward(ec_code)
            # 缓存结果（限制缓存大小）— detach to CPU to avoid pinning GPU memory
            if len(self._query_encoding_cache) < 100:
                self._query_encoding_cache[cache_key] = (ec_code.detach().cpu(), dg_features.detach().cpu())
        
        # ========== 3. CA3 模式补全召回 ==========
        # Build partial cue for CA3 recall
        partial_cue = {
            'features': dg_features,
            'semantic': query_semantic,
            'timestamp': query_timestamp
        }
        
        memories = self.ca3_memory.recall(
            query_features=partial_cue.get('features'),
            query_semantic=partial_cue.get('semantic'),
            query_timestamp=partial_cue.get('timestamp'),
            topk=topk * 5,  # Get more candidates
            return_all=True  # Skip internal tier threshold filtering
        )
        
        # ========== 4. 过滤低质量记忆（基于语义相关性，而非时间衰减）==========
        # BUG FIX: activation_strength 是时间衰减指标（从1.0开始随时间衰减），
        # 不是语义相关性指标。旧的 activation_strength < recall_threshold 过滤会
        # 错误地丢弃语义高度相关但只是较旧的记忆。
        # 现在改为：只过滤掉完全没有相关性信号的记忆。
        filtered_memories = []
        for mem in memories:
            # Only filter out memories with truly zero relevance signal
            has_similarity = hasattr(mem, '_embedding_score') and mem._embedding_score > 0.01
            has_keyword = hasattr(mem, '_recall_keyword_score') and mem._recall_keyword_score > 0
            has_dg_match = hasattr(mem, '_dg_match_score') and mem._dg_match_score > 0.01
            # Keep if ANY signal exists, OR if memory is core (never auto-filter core memories)
            if has_similarity or has_keyword or has_dg_match or getattr(mem, 'is_core', False):
                filtered_memories.append(mem)
            elif len(filtered_memories) < topk:
                # Keep some low-signal memories as fallback when we don't have enough
                filtered_memories.append(mem)
        memories = filtered_memories[:topk * 2]
        
        # ========== 5. 构建记忆字典并计算综合相关性分数 ==========
        memory_dg_features = {}
        memory_kv_features = {}
        memory_dicts = []
        current_timestamp = int(time.time() * 1000)
        
        for mem in memories:
            mem_dict = mem.to_dict()
            mem_id = mem_dict.get('memory_id', '')
            
            # --- 5a. 收集 DG 和 KV 特征 ---
            if mem.dg_features is not None:
                memory_dg_features[mem_id] = mem.dg_features
            if mem.key_features is not None:
                memory_kv_features[mem_id] = {
                    'key': mem.key_features,
                    'value': mem.value_features
                }
            
            # --- 5b. 确保关键字段存在 ---
            if 'content' not in mem_dict and mem_id in self.ca3_memory.memories:
                mem_dict['content'] = self.ca3_memory.memories[mem_id].content
            if 'is_core' not in mem_dict and mem_id in self.ca3_memory.memories:
                mem_dict['is_core'] = self.ca3_memory.memories[mem_id].is_core
            
            # Include semantic_pointer, key_entities, semantic_summary from the memory object
            if 'semantic_pointer' not in mem_dict and mem_id in self.ca3_memory.memories:
                mem_dict['semantic_pointer'] = self.ca3_memory.memories[mem_id].semantic_pointer
            if 'key_entities' not in mem_dict and hasattr(self.ca3_memory.memories[mem_id], 'key_entities'):
                mem_dict['key_entities'] = self.ca3_memory.memories[mem_id].key_entities
            if 'semantic_summary' not in mem_dict and hasattr(self.ca3_memory.memories[mem_id], 'semantic_summary'):
                mem_dict['semantic_summary'] = self.ca3_memory.memories[mem_id].semantic_summary
            
            # --- 5c. 计算综合相关性分数 ---
            embedding_score = getattr(mem, '_embedding_score', 0.0) or 0.0
            keyword_score = getattr(mem, '_recall_keyword_score', 0.0) or 0.0
            dg_match_score = getattr(mem, '_dg_match_score', 0.0) or 0.0
            activation = getattr(mem, 'activation_strength', 1.0) or 1.0
            is_core = getattr(mem, 'is_core', False)
            
            # Tier bonus: long-term memories get a small boost
            tier = getattr(mem, 'tier', 'short_term')
            # Handle both MemoryTier enum and string
            if hasattr(tier, 'name'):
                tier = tier.name.lower()
            tier_bonus = {'short_term': 0.0, 'mid_term': 0.05, 'long_term': 0.1}.get(tier, 0.0)
            
            # Core bonus
            core_bonus = 0.15 if is_core else 0.0
            
            # Combined relevance: weighted sum of all signals
            relevance_score = (
                0.4 * embedding_score +
                0.3 * min(keyword_score / 10.0, 1.0) +  # normalize keyword score
                0.2 * dg_match_score +
                tier_bonus +
                core_bonus
            )
            # Small decay factor so very old but irrelevant memories don't rank high
            relevance_score *= (0.5 + 0.5 * activation)
            
            mem_dict['relevance_score'] = round(relevance_score, 6)
            mem_dict['_activation_strength'] = activation
            
            memory_dicts.append(mem_dict)
        
        # --- 5d. 附加 DG 和 KV 特征到字典 ---
        for mem_dict in memory_dicts:
            mem_id = mem_dict.get('memory_id', '')
            
            # DG 特征（统一转 float32 避免 BFloat16 序列化错误）
            if mem_id in memory_dg_features:
                mem_dict['dg_features'] = memory_dg_features[mem_id].detach().cpu().float().numpy().tolist()
            else:
                mem_dict['dg_features'] = dg_features.detach().cpu().float().numpy().tolist()
            
            # KV 特征（用于窄带宽注意力）
            if mem_id in memory_kv_features:
                kv = memory_kv_features[mem_id]
                mem_dict['key_features'] = kv['key'].detach().cpu().float().numpy().tolist() if kv['key'] is not None else None
                mem_dict['value_features'] = kv['value'].detach().cpu().float().numpy().tolist() if kv['value'] is not None else None
        
        # ========== 6. 按综合相关性分数排序（替代纯时序排序）==========
        memory_dicts.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        
        # CA1 时序重排作为辅助信号（混合相关性+时序）
        if len(memory_dicts) > 1:
            try:
                temporal_sorted = self.ca1_gate.sort_by_temporal(
                    memories=memory_dicts,
                    current_timestamp=current_timestamp,
                    topk=topk
                )
                # Merge: use CA1 temporal order but respect relevance filtering
                # Keep only memories that passed our relevance filter
                temporal_ids = {m.get('memory_id') for m in temporal_sorted}
                # Prioritize temporal order but include high-relevance memories too
                final = []
                for m in temporal_sorted:
                    final.append(m)
                # Add any high-relevance memories missed by temporal sort
                final_ids = {m.get('memory_id') for m in final}
                for m in memory_dicts:
                    if m.get('memory_id') not in final_ids and m.get('relevance_score', 0) > 0.3:
                        final.append(m)
                memory_dicts = final
            except Exception as e:
                logger.debug(f"[Hippocampus] CA1 temporal sort skipped: {e}")
        
        # ========== 7. 记忆重构：用重构引擎丰富召回结果 ==========
        if self.reconstruction_engine is not None and query_semantic:
            try:
                recalled_memory_objects = []
                for mem_dict in memory_dicts[:topk]:
                    mem_id = mem_dict.get('memory_id', '')
                    if mem_id in self.ca3_memory.memories:
                        recalled_memory_objects.append(self.ca3_memory.memories[mem_id])
                if recalled_memory_objects:
                    reconstructed = self.reconstruction_engine.reconstruct_memory(
                        query=query_semantic,
                        relevant_memories=recalled_memory_objects,
                    )
                    if reconstructed and reconstructed.narrative:
                        if memory_dicts:
                            memory_dicts[0]['reconstructed_narrative'] = reconstructed.narrative
                            memory_dicts[0]['reconstruction_confidence'] = reconstructed.overall_confidence
            except Exception as e:
                logger.debug(f"[Hippocampus] 记忆重构失败: {e}")

        return memory_dicts[:topk]

    def get_state(self) -> dict:
        """获取海马体系统的完整状态"""
        return {
            'ca3_state': self.ca3_memory.get_state(),
            'ec_encoder_state': self.ec_encoder.state_dict(),
            'dg_separator_state': self.dg_separator.state_dict(),
            'ca1_gate_state': self.ca1_gate.state_dict(),
            'cycle_count': self.cycle_count
        }

    def set_semantic_model(self, model_interface):
        """注入模型接口到语义引擎（延迟注入，因为模型在 HippocampusSystem 之后创建）"""
        if self.semantic_engine is not None:
            self.semantic_engine.model = model_interface
            self.semantic_engine.device = self.device
            # 同时更新 CA3 中的引用
            self.ca3_memory.semantic_engine = self.semantic_engine
            logger.info("[Hippocampus] 语义引擎已注入模型引用")

    def set_state(self, state: dict):
        """从状态字典恢复海马体系统"""
        if 'ca3_state' in state:
            self.ca3_memory.set_state(state['ca3_state'])
        if 'ec_encoder_state' in state:
            self.ec_encoder.load_state_dict(state['ec_encoder_state'])
        if 'dg_separator_state' in state:
            self.dg_separator.load_state_dict(state['dg_separator_state'])
        if 'ca1_gate_state' in state:
            self.ca1_gate.load_state_dict(state['ca1_gate_state'])
        self.cycle_count = state.get('cycle_count', 0)

    def generate_attention_gate(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        memory_anchors: List[dict]
    ) -> torch.Tensor:
        """
        生成注意力门控信号
        
        Args:
            query: Query 向量 [batch, seq, hidden]
            key: Key 向量 [batch, seq, hidden]
            memory_anchors: 记忆锚点列表
        
        Returns:
            gate_mask: 注意力门控掩码
        """
        return self.ca1_gate.forward(query, key, memory_anchors)
    
    def update_memory_strength(self, memory_id: str, delta: float):
        """
        更新记忆激活强度 (STDP 更新接口)
        
        Args:
            memory_id: 记忆 ID
            delta: 强度变化量
        """
        self.ca3_memory.update_memory_strength(memory_id, delta)
    
    def prune_weak_memories(self, threshold: float = 0.3) -> int:
        """
        修剪弱记忆
        
        Args:
            threshold: 强度阈值
        
        Returns:
            pruned_count: 修剪的记忆数量
        """
        count = self.ca3_memory.prune_weak_memories(threshold)
        return count
    
    def _calculate_current_usage(self) -> int:
        """计算当前内存使用的辅助方法"""
        import sys
        total_size = 0
        
        # 计算CA3记忆的实际内存占用
        for memory_id, memory in self.ca3_memory.memories.items():
            # 基础数据结构大小
            memory_size = sys.getsizeof(memory_id)
            
            # 特征向量大小 - dg_features 是 EpisodicMemory 的可选属性
            if memory.dg_features is not None:
                memory_size += memory.dg_features.element_size() * memory.dg_features.nelement()
            
            # 文本数据大小 - semantic_pointer 是 EpisodicMemory 的必需属性
            memory_size += sys.getsizeof(memory.semantic_pointer)
            
            # context 是可选属性
            if hasattr(memory, 'context'):
                memory_size += sys.getsizeof(str(memory.context))
            
            total_size += memory_size
        
        # 加上EC和DG组件的内存 - 在 __init__ 中已初始化
        for param in self.ec_encoder.parameters():
            total_size += param.element_size() * param.nelement()
        
        for param in self.dg_separator.parameters():
            total_size += param.element_size() * param.nelement()
        
        return total_size

    def _update_memory_usage(self):
        """增量更新内存使用统计"""
        # 只在有新记忆时才更新
        current_count = len(self.ca3_memory.memories)
        # _last_memory_count 在 __init__ 中初始化为 0
        if current_count > self._last_memory_count:
            # 增量计算新增记忆的大小
            delta = current_count - self._last_memory_count
            avg_size = 1000  # 估算每个记忆的平均大小（字节）
            self.memory_usage_bytes += delta * avg_size
            self._last_memory_count = current_count
        
        # 每 100 次操作后做一次精确统计
        if self.cycle_count % 100 == 0:
            self.memory_usage_bytes = self._calculate_current_usage()
        
        # 检查是否超出限制
        if self.memory_usage_bytes > self.max_memory_bytes:
            excess_ratio = 1 - (self.max_memory_bytes / self.memory_usage_bytes)
            threshold = 0.3 + excess_ratio * 0.5
            self.prune_weak_memories(threshold)
            self.memory_usage_bytes = self._calculate_current_usage()
            self._last_memory_count = len(self.ca3_memory.memories)
    
    def start_swr_monitoring(self):
        """启动 SWR 离线回放监控"""
        self.swr_consolidation.start_monitoring()
    
    def stop_swr_monitoring(self):
        """停止 SWR 监控"""
        self.swr_consolidation.stop_monitoring()
    
    def trigger_swr_consolidation(self):
        """手动触发海马体 SWR 记忆巩固"""
        # swr_consolidation 在 __init__ 中已初始化
        self.swr_consolidation.trigger_manual_consolidation()

    def record_activity(self):
        """记录用户活动 (重置空闲计时器)"""
        self.swr_consolidation.record_activity()
        if self.dream_system is not None:
            self.dream_system.record_activity()
    
    def add_replay_sequence(
        self,
        sequence_id: str,
        memories: List[dict],
        reward_signal: float
    ):
        """添加回放序列"""
        self.swr_consolidation.add_replay_sequence(
            sequence_id=sequence_id,
            memories=memories,
            reward_signal=reward_signal
        )
    
    def forward(
        self,
        features: torch.Tensor,
        mode: str = "recall",
        **kwargs
    ):
        """
        前向传播
        
        Args:
            features: 输入特征
            mode: 执行模式 ("encode" | "recall" | "gate")
            **kwargs: 额外参数
        
        Returns:
            根据模式返回不同结果
        """
        if mode == "encode":
            return self.encode(
                features=features,
                token_id=kwargs.get('token_id', 0),
                timestamp=kwargs.get('timestamp', int(time.time() * 1000)),
                context=kwargs.get('context'),
                kv_features=kwargs.get('kv_features')
            )
        
        elif mode == "recall":
            return self.recall(
                query_features=features,
                topk=kwargs.get('topk', 2),
                query_semantic=kwargs.get('query_semantic'),
                query_timestamp=kwargs.get('query_timestamp')
            )
        
        elif mode == "gate":
            return self.generate_attention_gate(
                query=kwargs.get('query'),
                key=kwargs.get('key'),
                memory_anchors=features  # 这里 features 是记忆锚点
            )
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def consolidate_memories(self) -> dict:
        """执行记忆固化和衰减（供外部调用）"""
        return self.ca3_memory.consolidate_and_decay()

    def get_stats(self) -> dict:
        """获取系统统计信息（含分层统计）"""
        return {
            'cycle_count': self.cycle_count,
            'num_memories': len(self.ca3_memory.memories),
            'memory_usage_mb': self.memory_usage_bytes / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'ca3_stats': self.ca3_memory.get_stats(),
            'swr_stats': self.swr_consolidation.get_stats(),
            'device': self.device,
        }
    
    def _ensure_cpu_tensor(self, value_features):
        """Ensure value_features is a detached CPU tensor."""
        if isinstance(value_features, list):
            return torch.tensor(value_features, dtype=torch.float32).detach().cpu()
        elif isinstance(value_features, torch.Tensor):
            return value_features.detach().cpu()
        return value_features

    # ========== 联想记忆网络扩展方法 ==========

    def get_associated_memories(self, memory_id: str, max_depth: int = 2):
        """
        获取与指定记忆关联的所有记忆（BFS遍历关联图）

        Args:
            memory_id: 起始记忆ID
            max_depth: BFS最大深度（1=直接关联，2=二阶关联）

        Returns:
            关联记忆列表 [(memory_id, depth, cumulative_strength), ...]
        """
        if self.associative_network is None:
            return []
        return self.associative_network.get_associated_memories(memory_id, max_depth=max_depth)

    def spread_activation(self, memory_ids: list, iterations: int = 3):
        """
        扩散激活算法 —— 模拟人脑联想思维

        Args:
            memory_ids: 种子记忆ID列表
            iterations: 扩散迭代次数

        Returns:
            {memory_id: activation_value} 按激活值降序排列
        """
        if self.associative_network is None:
            return {}
        return self.associative_network.spread_activation(memory_ids, iterations=iterations)

    def reconstruct_recall(self, query: str, memories: list):
        """
        使用记忆重构引擎对召回结果进行重构式丰富

        Args:
            query: 查询文本
            memories: 记忆对象列表

        Returns:
            ReconstructedMemory 重构后的记忆对象，或 None
        """
        if self.reconstruction_engine is None or not memories:
            return None
        try:
            return self.reconstruction_engine.reconstruct_memory(
                query=query, relevant_memories=memories
            )
        except Exception as e:
            logger.debug(f"[Hippocampus] 记忆重构失败: {e}")
            return None

    def trigger_dream_consolidation(self):
        """
        手动触发梦境巩固系统

        Returns:
            DreamSequence 梦境记录，或 None
        """
        if self.dream_system is None:
            return None
        try:
            return self.dream_system.trigger_sleep(depth=0.5, num_cycles=4)
        except Exception as e:
            logger.warning(f"[Hippocampus] 梦境巩固触发失败: {e}")
            return None

    def reset(self):
        """重置系统状态"""
        self.cycle_count = 0
        self.ca3_memory.memories.clear()
        self.ca3_memory.time_index.clear()
        self.ca3_memory.semantic_index.clear()
        self._query_encoding_cache.clear()
        self._last_memory_count = 0
        self.memory_usage_bytes = 0
        # 重置联想记忆网络状态
        if self.associative_network is not None:
            self.associative_network.adjacency_graph.clear()
            self.associative_network._reverse_index.clear()
        self._update_memory_usage()
    
    # ========== KV Cache 专用方法 (用于滑动窗口管理) ==========
    
    def store_kv_as_memory(
        self,
        kv_features: Dict[str, Any],
        context_text: str
    ) -> str:
        """
        将KV cache作为记忆存储
        
        用于滑动窗口管理器，将被释放的KV存储到海马体
        
        Args:
            kv_features: KV特征字典，包含:
                - key_features: Key特征向量
                - value_features: Value特征向量
                - num_layers: 层数
                - seq_len: 序列长度
                - timestamp: 时间戳
            context_text: 上下文文本
        
        Returns:
            memory_id: 生成的记忆ID
        """
        # 提取KV特征
        key_features = kv_features.get('key_features', [])
        value_features = kv_features.get('value_features', [])
        num_layers = kv_features.get('num_layers', 0)
        seq_len = kv_features.get('seq_len', 0)
        timestamp = kv_features.get('timestamp', int(time.time() * 1000))
        
        if not key_features or not value_features:
            logger.warning("[Hippocampus] KV特征为空，跳过存储")
            return ""
        
        # 使用KV特征作为记忆特征
        # 将特征向量转换为tensor
        key_tensor = torch.tensor(key_features, dtype=torch.float32, device=self.device)
        
        if key_tensor.dim() == 1:
            key_tensor = key_tensor.unsqueeze(0)
        
        # 使用key特征作为记忆编码的输入
        # EC编码
        ec_code = self.ec_encoder.encode_single(key_tensor.squeeze(0))
        
        # DG模式分离
        dg_output, memory_id = self.dg_separator.separate_and_id(ec_code)
        
        # 构建记忆上下文
        context = [{
            'content': context_text,
            'type': 'kv_memory',
            'num_layers': num_layers,
            'seq_len': seq_len,
            'is_core': False,
            'semantic_pointer': f"kv_{timestamp}"
        }]
        
        # 存储到CA3
        self.ca3_memory.store(
            memory_id=memory_id,
            timestamp=timestamp,
            semantic_pointer=f"kv_{timestamp}",
            temporal_skeleton="",
            causal_links=[],
            dg_features=dg_output.detach().cpu(),
            is_core=False,
            content=context_text,
            key_features=key_tensor.detach().cpu() if isinstance(key_tensor, torch.Tensor) else key_features,
            value_features=self._ensure_cpu_tensor(value_features)
        )
        
        logger.debug(
            f"[Hippocampus] KV已存储为记忆: "
            f"memory_id={memory_id}, "
            f"seq_len={seq_len}, "
            f"layers={num_layers}"
        )
        
        return memory_id
    
    def recall_kv(
        self,
        query_features: torch.Tensor,
        topk: int = 2
    ) -> List[Dict[str, Any]]:
        """
        召回相关的KV记忆
        
        用于滑动窗口管理器，从海马体检索相关的KV特征
        
        Args:
            query_features: 查询特征
            topk: 返回的记忆数量
        
        Returns:
            kv_memories: KV记忆列表，每个元素包含:
                - memory_id: 记忆ID
                - kv_features: KV特征字典
                - content: 上下文文本
                - similarity: 相似度
        """
        # 使用常规召回方法
        memories = self.recall(
            query_features=query_features,
            topk=topk * 2  # 多召回一些，后续过滤
        )
        
        # 过滤出包含KV特征的记忆
        kv_memories = []
        for mem in memories:
            # 修复: 检查 key_features 字段（to_dict() 序列化为 'key_features'）
            # 之前的条件 mem.get('type')=='kv_memory' 和 'kv_features' in mem 永远不匹配
            if mem.get('key_features') is not None or mem.get('value_features') is not None:
                kv_memories.append({
                    'memory_id': mem.get('memory_id', ''),
                    'kv_features': {
                        'key_features': mem.get('key_features'),
                        'value_features': mem.get('value_features'),
                        'num_layers': mem.get('num_layers', 0),
                        'seq_len': mem.get('seq_len', 0)
                    },
                    'content': mem.get('content', ''),
                    'similarity': mem.get('activation_strength', 0.0)
                })
        
        # 返回topk个
        return kv_memories[:topk]
