"""
海马体记忆系统 - 完整集成

将 EC、DG、CA3、CA1、SWR 五个子模块集成为完整的类脑海马体系统
提供统一的编码、召回、门控接口
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import time

from .ec_encoder import EntorhinalEncoder
from .dg_separator import DentateGyrusSeparator
from .ca3_memory import CA3EpisodicMemory
from .ca1_gate import CA1AttentionGate
from .swr_consolidation import SWRConsolidation


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
    
    def __init__(self, config, device: Optional[str] = None, hidden_size: int = 1024):
        super().__init__()
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size  # 动态适配模型隐藏层大小
        
        # 从配置加载参数
        hc_config = config.hippocampus
        
        # ========== 1. EC 内嗅皮层 - 特征编码 ==========
        self.ec_encoder = EntorhinalEncoder(
            input_dim=hidden_size,         # 动态适配模型隐藏层大小
            output_dim=hc_config.EC_feature_dim,
            sparsity=hc_config.DG_sparsity,
            freeze_encoder=True
        )
        
        # ========== 2. DG 齿状回 - 模式分离 ==========
        self.dg_separator = DentateGyrusSeparator(
            input_dim=hc_config.EC_feature_dim,
            output_dim=hc_config.EC_feature_dim * 2,  # 扩展到 128 维
            sparsity=hc_config.DG_sparsity,
            orthogonalization=hc_config.DG_orthogonalization
        )
        
        # ========== 3. CA3 情景记忆库 ==========
        self.ca3_memory = CA3EpisodicMemory(
            max_capacity=hc_config.CA3_max_capacity,
            feature_dim=hc_config.EC_feature_dim * 2,
            timestamp_precision_ms=hc_config.CA3_timestamp_precision_ms,
            recall_threshold=0.7,
            decay_rate=0.999
        )
        
        # ========== 4. CA1 注意力 gate ==========
        self.ca1_gate = CA1AttentionGate(
            feature_dim=hc_config.EC_feature_dim * 2,
            hidden_size=hidden_size,        # 动态适配模型隐藏层大小
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
        
        # ========== 内存监控 ==========
        self.max_memory_bytes = hc_config.max_memory_bytes
        self.memory_usage_bytes = 0
        
        # ========== 周期计数器 ==========
        self.cycle_count = 0
    
    def encode(
        self,
        features: torch.Tensor,
        token_id: int,
        timestamp: int,
        context: Optional[List[dict]] = None
    ) -> str:
        """
        记忆编码流程
        
        完整流程：Token 特征 → EC 编码 → DG 分离 → CA3 存储
        
        Args:
            features: Token 特征 [hidden_size] 或 [1, hidden_size]
            token_id: Token ID
            timestamp: 时间戳 (ms)
            context: 上下文信息
        
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
        
        if context and len(context) > 0:
            # 从前序上下文提取时序关系
            prev_tokens = [ctx.get('token_id', -1) for ctx in context]
            temporal_skeleton = f"{prev_tokens} -> {token_id}"
            
            # 简单因果推断 (实际应更复杂)
            causal_links.append(f"preceded_by_{prev_tokens[-1]}" if prev_tokens else "start")
        
        # ========== 5. CA3 存储 ==========
        # 优先使用context中的semantic_pointer，否则使用token_id
        semantic_pointer = f"token_{token_id}"
        if context and len(context) > 0:
            # 从context中提取semantic_pointer
            for ctx in context:
                if 'semantic_pointer' in ctx and ctx['semantic_pointer']:
                    semantic_pointer = ctx['semantic_pointer']
                    break
        
        self.ca3_memory.store(
            memory_id=memory_id,
            timestamp=timestamp,
            semantic_pointer=semantic_pointer,
            temporal_skeleton=temporal_skeleton,
            causal_links=causal_links,
            dg_features=dg_output.detach().cpu()
        )
        
        # ========== 6. 更新内存使用 ==========
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
        记忆召回流程
        
        完整流程：查询特征 → CA3 模式补全 → CA1 时序排序 → 返回锚点
        
        Args:
            query_features: 查询特征 [hidden_size] 或 [1, hidden_size]
            topk: 返回数量
            query_semantic: 语义线索 (可选)
            query_timestamp: 时间线索 (可选)
        
        Returns:
            memory_anchors: 召回的记忆锚点列表
        """
        # ========== 1. 特征预处理 ==========
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        
        # ========== 2. EC 编码 (复用编码路径) ==========
        ec_code = self.ec_encoder.encode_single(query_features.squeeze(0))
        
        # ========== 3. DG 分离 ==========
        dg_features = self.dg_separator.forward(ec_code)
        
        # ========== 4. CA3 模式补全召回 ==========
        memories = self.ca3_memory.complete_pattern(
            partial_cue={
                'features': dg_features,
                'semantic': query_semantic,
                'timestamp': query_timestamp
            },
            topk=topk * 2  # 多召回一些，后续排序
        )
        
        # ========== 5. CA1 时序排序 ==========
        current_timestamp = int(time.time() * 1000)
        sorted_memories = self.ca1_gate.sort_by_temporal(
            memories=[mem.to_dict() for mem in memories],
            current_timestamp=current_timestamp,
            topk=topk
        )
        
        # ========== 6. 添加记忆本身的 DG 特征到返回结果 ==========
        # 创建memory_id到dg_features的映射
        memory_features = {}
        for mem in memories:
            if hasattr(mem, 'dg_features') and mem.dg_features is not None:
                memory_features[mem.memory_id] = mem.dg_features
        
        for mem_dict in sorted_memories:
            mem_id = mem_dict.get('memory_id', '')
            if mem_id in memory_features:
                mem_dict['dg_features'] = memory_features[mem_id].cpu()
            else:
                # 如果没有找到，使用查询的dg_features作为回退
                mem_dict['dg_features'] = dg_features.cpu()
        
        return sorted_memories

    def get_state(self) -> dict:
        """获取海马体系统的完整状态"""
        return {
            'ca3_state': self.ca3_memory.get_state(),
            'ec_encoder_state': self.ec_encoder.state_dict(),
            'dg_separator_state': self.dg_separator.state_dict(),
            'ca1_gate_state': self.ca1_gate.state_dict(),
            'cycle_count': self.cycle_count
        }

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
        self._update_memory_usage()
        return count
    
    def _update_memory_usage(self):
        """更新内存使用统计"""
        # 估算内存使用 (简化计算)
        num_memories = len(self.ca3_memory.memories)
        avg_memory_size = 1024  # 每个记忆约 1KB
        
        self.memory_usage_bytes = num_memories * avg_memory_size
        
        # 检查是否超出限制
        if self.memory_usage_bytes > self.max_memory_bytes:
            # 自动修剪
            excess_ratio = 1 - (self.max_memory_bytes / self.memory_usage_bytes)
            threshold = 0.3 + excess_ratio * 0.5
            self.prune_weak_memories(threshold)
    
    def start_swr_monitoring(self):
        """启动 SWR 离线回放监控"""
        self.swr_consolidation.start_monitoring()
    
    def stop_swr_monitoring(self):
        """停止 SWR 监控"""
        self.swr_consolidation.stop_monitoring()
    
    def record_activity(self):
        """记录用户活动 (重置空闲计时器)"""
        self.swr_consolidation.record_activity()
    
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
                context=kwargs.get('context')
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
    
    def get_stats(self) -> dict:
        """获取系统统计信息"""
        return {
            'cycle_count': self.cycle_count,
            'num_memories': len(self.ca3_memory.memories),
            'memory_usage_mb': self.memory_usage_bytes / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'ca3_stats': self.ca3_memory.get_stats(),
            'swr_stats': self.swr_consolidation.get_stats(),
            'device': self.device
        }
    
    def reset(self):
        """重置系统状态"""
        self.cycle_count = 0
        self.ca3_memory.memories.clear()
        self.ca3_memory.time_index.clear()
        self.ca3_memory.semantic_index.clear()
        self._update_memory_usage()
