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
import torch.nn.functional as F
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
    is_core: bool = False             # 核心记忆标记
    content: str = ""                 # 完整内容
    dg_features: Optional[torch.Tensor] = None  # DG 分离后的特征
    # ========== 新增: 用于窄带宽注意力的 KV 特征 ==========
    key_features: Optional[torch.Tensor] = None    # [num_heads, head_dim]
    value_features: Optional[torch.Tensor] = None  # [num_heads, head_dim]
    
    def to_dict(self) -> dict:
        """转换为字典 - 避免序列化torch.Tensor"""
        result = {
            'memory_id': self.memory_id,
            'timestamp': self.timestamp,
            'temporal_skeleton': self.temporal_skeleton,
            'semantic_pointer': self.semantic_pointer,
            'causal_links': self.causal_links,
            'activation_strength': self.activation_strength,
            'is_core': self.is_core,
            'content': self.content
        }
        
        # 安全序列化tensor：detach并转为numpy
        if self.dg_features is not None:
            try:
                result['dg_features'] = self.dg_features.detach().cpu().numpy().tolist()
            except:
                result['dg_features'] = None
        
        # 序列化 KV 特征
        if self.key_features is not None:
            try:
                result['key_features'] = self.key_features.detach().cpu().numpy().tolist()
            except:
                result['key_features'] = None
        
        if self.value_features is not None:
            try:
                result['value_features'] = self.value_features.detach().cpu().numpy().tolist()
            except:
                result['value_features'] = None
        
        return result


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
        dg_features: Optional[torch.Tensor] = None,
        is_core: bool = False,
        content: str = "",
        key_features: Optional[torch.Tensor] = None,      # 新增: 用于窄带宽注意力
        value_features: Optional[torch.Tensor] = None     # 新增: 用于窄带宽注意力
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
            is_core: 是否为核心记忆
            content: 完整内容
            key_features: 注意力 Key 特征 [num_heads, head_dim]
            value_features: 注意力 Value 特征 [num_heads, head_dim]
        
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
            dg_features=dg_features,
            is_core=is_core,
            content=content,
            activation_strength=10.0 if is_core else 1.0,  # 核心记忆初始激活强度更高
            key_features=key_features,      # 新增
            value_features=value_features   # 新增
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
        
        # ========== 1. 优先召回核心记忆（关键词匹配） ==========
        core_memories = [m for m in self.memories.values() if m.is_core]
        
        # 如果有语义线索，优先匹配核心记忆的关键词
        if query_semantic:
            # 关键词匹配
            keywords = query_semantic.lower().split()
            for memory in core_memories:
                # 检查语义指针或内容是否包含关键词
                semantic_text = (memory.semantic_pointer + " " + memory.content).lower()
                if any(kw in semantic_text for kw in keywords):
                    candidates.append(memory)
        
        # ========== 2. 特征相似度匹配 ==========
        if query_features is not None and len(candidates) < topk:
            # 收集所有记忆的特征
            all_features = []
            all_ids = []
            
            for mid, memory in self.memories.items():
                if memory.dg_features is not None:
                    all_features.append(memory.dg_features)
                    all_ids.append(mid)
            
            if all_features:
                all_features = torch.stack(all_features)
                
                if all_features.numel() > 0:
                    # 确保query_features是二维的
                    if query_features.dim() == 1:
                        query_features = query_features.unsqueeze(0)
                    
                    # 批量计算余弦相似度
                    query_norm = F.normalize(query_features, p=2, dim=-1)
                    all_features_norm = F.normalize(all_features, p=2, dim=-1)
                    
                    # 确保维度匹配
                    if query_norm.dim() == 2 and all_features_norm.dim() == 2:
                        similarities = torch.mm(query_norm, all_features_norm.t()).squeeze(0)
                    else:
                        # 回退到逐个计算
                        similarities = torch.zeros(1, all_features_norm.shape[0])
                        for i, feat in enumerate(all_features_norm):
                            similarities[0, i] = F.cosine_similarity(query_norm.flatten().unsqueeze(0), feat.flatten().unsqueeze(0))
                    
                    # 获取 Top-K 相似的记忆
                    # 核心记忆使用更低的阈值
                    top_sim, top_indices = torch.topk(similarities, k=min(topk * 2, len(all_ids)))
                    
                    for i, sim in enumerate(top_sim):
                        memory_id = all_ids[top_indices[i]]
                        memory = self.memories[memory_id]
                        
                        # 核心记忆使用更低的召回阈值
                        threshold = 0.4 if memory.is_core else 0.5  # 降低阈值提高召回率
                        
                        if sim > threshold:
                            candidates.append(memory)
        
        # ========== 3. 如果召回结果不足，直接返回核心记忆 ==========
        if len(candidates) < topk and core_memories:
            for memory in core_memories:
                if memory not in candidates:
                    candidates.append(memory)
                    if len(candidates) >= topk:
                        break
        
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
    
    def get_state(self) -> dict:
        """获取 CA3 模块的完整状态 - 修复版：安全序列化"""
        # 将所有EpisodicMemory对象转换为字典
        memories_dict = {}
        for mem_id, memory in self.memories.items():
            memories_dict[mem_id] = memory.to_dict()  # 使用to_dict()方法安全转换
        
        return {
            'memories': memories_dict,
            'time_index': self.time_index,
            'semantic_index': self.semantic_index,
            'current_timestamp': self.current_timestamp
        }
    
    def set_state(self, state: dict):
        """从状态字典恢复 CA3 模块 - 修复版：正确处理字典数据"""
        memories_dict = state.get('memories', {})
        
        # 将字典转换回EpisodicMemory对象
        self.memories = OrderedDict()
        for mem_id, mem_dict in memories_dict.items():
            # 处理dg_features字段
            dg_features = None
            if 'dg_features' in mem_dict and mem_dict['dg_features'] is not None:
                try:
                    # 如果是list，转换为tensor
                    if isinstance(mem_dict['dg_features'], list):
                        dg_features = torch.tensor(mem_dict['dg_features'], dtype=torch.float32)
                    elif isinstance(mem_dict['dg_features'], torch.Tensor):
                        dg_features = mem_dict['dg_features']
                except:
                    dg_features = None
            
            # 创建EpisodicMemory对象
            memory = EpisodicMemory(
                memory_id=mem_dict.get('memory_id', mem_id),
                timestamp=mem_dict.get('timestamp', 0),
                temporal_skeleton=mem_dict.get('temporal_skeleton', ''),
                semantic_pointer=mem_dict.get('semantic_pointer', ''),
                causal_links=mem_dict.get('causal_links', []),
                activation_strength=mem_dict.get('activation_strength', 1.0),
                is_core=mem_dict.get('is_core', False),
                content=mem_dict.get('content', ''),
                dg_features=dg_features
            )
            self.memories[mem_id] = memory
        
        self.time_index = state.get('time_index', {})
        self.semantic_index = state.get('semantic_index', {})
        self.current_timestamp = state.get('current_timestamp', 0)
    
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
        
        优化:
        - 添加衰减机制，防止过度增强
        - 添加时间衰减因子
        
        Args:
            memory_id: 记忆 ID
            delta: 强度变化量
        """
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            
            # 时间衰减因子：越久的记忆衰减越慢（长期记忆固化）
            time_elapsed = (self.current_timestamp - memory.timestamp) / 1000.0  # 秒
            time_decay = 1.0 / (1.0 + time_elapsed * 0.001)  # 轻微衰减
            
            # 应用更新
            effective_delta = delta * time_decay
            memory.activation_strength += effective_delta
            
            # 限制范围 [0.1, 2.0]，最低0.1防止完全遗忘
            memory.activation_strength = max(0.1, min(2.0, memory.activation_strength))
    
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
