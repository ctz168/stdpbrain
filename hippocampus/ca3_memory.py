"""
CA3 区 - 情景记忆库 + 模式补全单元

功能:
- 以「记忆 ID+10ms 级时间戳 + 时序骨架 + 语义指针 + 因果关联」格式存储情景记忆
- 仅存指针不存完整文本 (节省内存)
- 基于部分线索完成完整记忆链条召回
- 每个周期输出 1-2 个最相关记忆锚点
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
from collections import OrderedDict


@dataclass
class EpisodicMemory:
    """情景记忆数据结构"""
    memory_id: str                    # 唯一记忆 ID(DG 正交化生成)
    timestamp: int                    # 10ms 级时间戳
    temporal_skeleton: str            # 时序骨架 (前后 token 关系)
    semantic_pointer: str             # 语义指针 (不存完整文本)
    causal_links: List[str]           # 因果关联列表
    activation_strength: float = 1.0  # 激活强度 (STDP 权重)
    dg_features: Optional[torch.Tensor] = None  # DG 分离后的特征
    
    def to_dict(self) -> dict:
        return {
            'memory_id': self.memory_id,
            'timestamp': self.timestamp,
            'temporal_skeleton': self.temporal_skeleton,
            'semantic_pointer': self.semantic_pointer,
            'causal_links': self.causal_links,
            'activation_strength': self.activation_strength
        }


class CA3EpisodicMemory(nn.Module):
    """
    CA3 情景记忆库
    
    使用循环缓存管理记忆存储，支持:
    - 快速存储与检索
    - 模式补全 (基于部分线索召回完整记忆)
    - 记忆修剪 (遗忘弱记忆)
    """
    
    def __init__(
        self,
        max_capacity: int = 10000,     # 最大记忆容量
        feature_dim: int = 128,        # DG 特征维度
        timestamp_precision_ms: int = 10,  # 时间戳精度 10ms
        recall_threshold: float = 0.7, # 召回阈值
        decay_rate: float = 0.999      # 记忆衰减率
    ):
        super().__init__()
        self.max_capacity = max_capacity
        self.feature_dim = feature_dim
        self.timestamp_precision_ms = timestamp_precision_ms
        self.recall_threshold = recall_threshold
        self.decay_rate = decay_rate
        
        # ========== 记忆存储 (有序字典实现循环缓存) ==========
        self.memories: OrderedDict[str, EpisodicMemory] = OrderedDict()
        
        # ========== 索引结构 ==========
        # 时间戳索引：timestamp -> [memory_ids]
        self.time_index: Dict[int, List[str]] = {}
        
        # 语义指针索引：semantic_pointer -> memory_id
        self.semantic_index: Dict[str, str] = {}
        
        # ========== 当前时间戳 ==========
        self.current_timestamp = 0
    
    def store(
        self,
        memory_id: str,
        timestamp: int,
        semantic_pointer: str,
        temporal_skeleton: str = "",
        causal_links: List[str] = None,
        dg_features: Optional[torch.Tensor] = None
    ) -> EpisodicMemory:
        """
        存储情景记忆
        
        Args:
            memory_id: 唯一记忆 ID
            timestamp: 时间戳 (ms)
            semantic_pointer: 语义指针
            temporal_skeleton: 时序骨架
            causal_links: 因果关联
            dg_features: DG 特征
        
        Returns:
            memory: 存储的记忆对象
        """
        # ========== 1. 创建记忆对象 ==========
        memory = EpisodicMemory(
            memory_id=memory_id,
            timestamp=timestamp,
            temporal_skeleton=temporal_skeleton,
            semantic_pointer=semantic_pointer,
            causal_links=causal_links or [],
            dg_features=dg_features
        )
        
        # ========== 2. 检查容量，必要时删除最旧记忆 ==========
        if len(self.memories) >= self.max_capacity:
            oldest_id = next(iter(self.memories))
            self._remove_memory(oldest_id)
        
        # ========== 3. 存储记忆 ==========
        self.memories[memory_id] = memory
        
        # ========== 4. 更新索引 ==========
        # 时间戳索引
        ts_key = timestamp // self.timestamp_precision_ms
        if ts_key not in self.time_index:
            self.time_index[ts_key] = []
        self.time_index[ts_key].append(memory_id)
        
        # 语义索引
        self.semantic_index[semantic_pointer] = memory_id
        
        # ========== 5. 更新时间戳 ==========
        self.current_timestamp = max(self.current_timestamp, timestamp)
        
        return memory
    
    def recall(
        self,
        query_features: Optional[torch.Tensor] = None,
        query_semantic: Optional[str] = None,
        query_timestamp: Optional[int] = None,
        topk: int = 2
    ) -> List[EpisodicMemory]:
        """
        记忆召回
        
        Args:
            query_features: 查询特征 (用于相似度匹配)
            query_semantic: 语义线索
            query_timestamp: 时间线索
            topk: 返回数量
        
        Returns:
            memories: 召回的记忆列表
        """
        candidates = []
        
        # ========== 1. 语义线索检索 ==========
        if query_semantic and query_semantic in self.semantic_index:
            memory_id = self.semantic_index[query_semantic]
            if memory_id in self.memories:
                candidates.append(self.memories[memory_id])
        
        # ========== 2. 时间线索检索 ==========
        if query_timestamp is not None:
            ts_key = query_timestamp // self.timestamp_precision_ms
            # 查找相邻时间窗口的记忆
            for delta in [-1, 0, 1]:
                neighbor_ts = ts_key + delta
                if neighbor_ts in self.time_index:
                    for mem_id in self.time_index[neighbor_ts]:
                        if mem_id in self.memories:
                            candidates.append(self.memories[mem_id])
        
        # ========== 3. 特征相似度检索 ==========
        if query_features is not None:
            # 计算与所有记忆的 DG 特征相似度
            similarities = []
            for mem_id, memory in self.memories.items():
                if memory.dg_features is not None:
                    sim = self._compute_similarity(query_features, memory.dg_features)
                    if sim > self.recall_threshold:
                        candidates.append((sim, memory))
            
            # 按相似度排序
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = [mem for _, mem in candidates]
        
        # ========== 4. 去重 ==========
        seen_ids = set()
        unique_candidates = []
        for cand in candidates:
            if cand.memory_id not in seen_ids:
                seen_ids.add(cand.memory_id)
                unique_candidates.append(cand)
        
        # ========== 5. 按激活强度排序并返回 TopK ==========
        unique_candidates.sort(key=lambda m: m.activation_strength, reverse=True)
        return unique_candidates[:topk]
    
    def complete_pattern(
        self,
        partial_cue: dict,
        topk: int = 2
    ) -> List[EpisodicMemory]:
        """
        模式补全 - 基于部分线索召回完整记忆链条
        
        Args:
            partial_cue: 部分线索字典
                - semantic: 语义线索
                - timestamp: 时间线索
                - temporal: 时序线索
            topk: 返回数量
        
        Returns:
            memories: 补全后的记忆列表
        """
        # 从部分线索重构查询
        query_semantic = partial_cue.get('semantic')
        query_timestamp = partial_cue.get('timestamp')
        query_features = partial_cue.get('features')
        
        # 调用 recall
        return self.recall(
            query_features=query_features,
            query_semantic=query_semantic,
            query_timestamp=query_timestamp,
            topk=topk
        )
    
    def update_memory_strength(self, memory_id: str, delta: float):
        """
        更新记忆激活强度 (STDP 更新接口)
        
        Args:
            memory_id: 记忆 ID
            delta: 强度变化量
        """
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.activation_strength += delta
            memory.activation_strength = max(0.0, min(2.0, memory.activation_strength))
    
    def prune_weak_memories(self, threshold: float = 0.3):
        """
        修剪弱记忆 (遗忘机制)
        
        Args:
            threshold: 强度阈值，低于此值的记忆被删除
        """
        to_remove = []
        for memory_id, memory in self.memories.items():
            if memory.activation_strength < threshold:
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            self._remove_memory(memory_id)
        
        return len(to_remove)
    
    def _remove_memory(self, memory_id: str):
        """删除记忆及其索引"""
        if memory_id not in self.memories:
            return
        
        memory = self.memories[memory_id]
        
        # 删除时间戳索引
        ts_key = memory.timestamp // self.timestamp_precision_ms
        if ts_key in self.time_index and memory_id in self.time_index[ts_key]:
            self.time_index[ts_key].remove(memory_id)
        
        # 删除语义索引
        if memory.semantic_pointer in self.semantic_index:
            del self.semantic_index[memory.semantic_pointer]
        
        # 删除记忆本身
        del self.memories[memory_id]
    
    def _compute_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """计算特征相似度 (余弦相似度)"""
        feat1_flat = feat1.flatten()
        feat2_flat = feat2.flatten()
        
        if len(feat1_flat) != len(feat2_flat):
            return 0.0
        
        sim = torch.nn.functional.cosine_similarity(
            feat1_flat.unsqueeze(0),
            feat2_flat.unsqueeze(0),
            dim=1
        ).item()
        return sim
    
    def get_stats(self) -> dict:
        """获取记忆库统计信息"""
        if not self.memories:
            return {
                'num_memories': 0,
                'capacity_usage': 0.0,
                'avg_activation': 0.0
            }
        
        activations = [m.activation_strength for m in self.memories.values()]
        
        return {
            'num_memories': len(self.memories),
            'capacity_usage': len(self.memories) / self.max_capacity,
            'avg_activation': sum(activations) / len(activations),
            'max_activation': max(activations),
            'min_activation': min(activations)
        }
    
    def forward(
        self,
        query_features: torch.Tensor,
        topk: int = 2
    ) -> List[dict]:
        """前向传播 - 记忆召回"""
        memories = self.recall(query_features=query_features, topk=topk)
        return [mem.to_dict() for mem in memories]
