"""
CA1 区 - 时序编码 + 注意力门控单元

功能:
- 为每个记忆单元打精准时间戳，绑定时序 - 情景 - 因果关系
- 形成连续记忆链条
- 每个刷新周期输出 1-2 个最相关记忆锚点给模型注意力层
- 直接控制注意力聚焦方向
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MemoryAnchor:
    """记忆锚点数据结构"""
    memory_id: str
    timestamp: int
    temporal_context: str      # 时序上下文
    causal_chain: List[str]    # 因果链条
    attention_weight: float    # 注意力权重
    gate_signal: torch.Tensor  # 门控信号


class CA1AttentionGate(nn.Module):
    """
    CA1 注意力门控
    
    将 CA3 召回的记忆转换为注意力门控信号
    引导模型注意力聚焦到关键位置
    """
    
    def __init__(
        self,
        feature_dim: int = 128,        # DG 特征维度
        hidden_size: int = 2048,       # 模型 hidden size
        recall_topk: int = 2,          # 召回 TopK
        temporal_encoding: bool = True, # 启用时序编码
        gate_type: str = "additive"    # 门控类型："additive" | "multiplicative"
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.recall_topk = recall_topk
        self.temporal_encoding = temporal_encoding
        self.gate_type = gate_type
        
        # ========== 时序编码器 ==========
        if self.temporal_encoding:
            self.temporal_encoder = nn.Sequential(
                nn.Linear(1, feature_dim // 4),  # 时间戳→时序特征
                nn.ReLU(),
                nn.Linear(feature_dim // 4, feature_dim)
            )
        
        # ========== 注意力权重计算 ==========
        self.attention_scorer = nn.Sequential(
            nn.Linear(feature_dim + hidden_size, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1)
        )
        
        # ========== 门控投影 ==========
        if gate_type == "additive":
            self.gate_projection = nn.Linear(feature_dim, hidden_size)
        elif gate_type == "multiplicative":
            self.gate_projection = nn.Sequential(
                nn.Linear(feature_dim, hidden_size),
                nn.Sigmoid()  # 输出 [0, 1] 作为乘法门控
            )
    
    def forward(
        self,
        query: torch.Tensor,           # [batch, seq, hidden]
        key: torch.Tensor,             # [batch, seq, hidden]
        memory_anchors: List[dict]     # CA3 召回的记忆
    ) -> torch.Tensor:
        """
        生成注意力门控信号
        
        Args:
            query: Query 向量
            key: Key 向量
            memory_anchors: 记忆锚点列表
        
        Returns:
            gate_mask: 注意力门控掩码 [batch, heads, seq, seq]
        """
        batch_size, seq_len, _ = query.shape
        
        # ========== 1. 为空情况处理 ==========
        if not memory_anchors:
            return torch.zeros(
                batch_size, 1, seq_len, seq_len,
                device=query.device
            )
        
        # ========== 2. 为每个记忆锚点生成门控信号 ==========
        gate_signals = []
        for anchor in memory_anchors[:self.recall_topk]:
            gate_signal = self._generate_gate_signal(
                query=query,
                memory_data=anchor,
                key=key
            )
            gate_signals.append(gate_signal)
        
        # ========== 3. 聚合多个门控信号 ==========
        if len(gate_signals) > 0:
            # 取平均或加权平均
            aggregated_gate = torch.stack(gate_signals).mean(dim=0)
        else:
            aggregated_gate = torch.zeros(
                batch_size, 1, seq_len, seq_len,
                device=query.device
            )
        
        return aggregated_gate
    
    def _generate_gate_signal(
        self,
        query: torch.Tensor,
        memory_data: dict,
        key: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """为单个记忆锚点生成门控信号"""
        if key is None:
            key = query  # 如果没有提供key，使用query作为fallback
        
        batch_size, seq_len, hidden_size = query.shape
        
        # ========== 提取记忆特征 ==========
        if 'dg_features' in memory_data and memory_data['dg_features'] is not None:
            mem_features = memory_data['dg_features']
            # 确保类型匹配
            if isinstance(mem_features, torch.Tensor):
                mem_features = mem_features.to(dtype=query.dtype, device=query.device)
            else:
                mem_features = torch.tensor(mem_features, dtype=query.dtype, device=query.device)
        else:
            # 简化处理：随机特征 (匹配 query 的数据类型)
            mem_features = torch.randn(self.feature_dim, dtype=query.dtype, device=query.device)
        
        # ========== 时序编码 (可选) ==========
        if self.temporal_encoding and 'timestamp' in memory_data:
            timestamp = torch.tensor(
                [[memory_data['timestamp']]], 
                dtype=query.dtype,  # 匹配 query 的数据类型
                device=query.device
            )
            temporal_feature = self.temporal_encoder(timestamp)
            mem_features = mem_features + temporal_feature.squeeze(0)
        
        # ========== 计算 token-wise 注意力权重 ==========
        # 将记忆特征投影到 hidden_size 空间，以便与 query 进行相似度计算
        mem_proj = self.gate_projection(mem_features.unsqueeze(0)) # [1, hidden_size]
        
        # 计算 query 与 记忆特征 的余弦相似度作为权重
        # query: [batch, seq_len, hidden_size]
        query_norm = F.normalize(query, p=2, dim=-1)
        mem_norm = F.normalize(mem_proj, p=2, dim=-1)
        
        # attention_weight: [batch, seq_len, 1] - 每个 token 对该记忆的相关度
        attention_weight = torch.matmul(query_norm, mem_norm.transpose(-2, -1))
        attention_weight = torch.sigmoid(attention_weight * 5.0) # 放大差异
        
        # ========== 生成门控掩码 ==========
        if self.gate_type == "additive":
            # 加法门控：生成偏置项
            # 直接使用投影后的记忆特征与 key 的交互
            gate_bias = torch.matmul(mem_proj, key.transpose(-2, -1)) # [batch, 1, seq_len]
            
            # gate_mask: [batch, 1, seq_len, seq_len]
            # 这里对应于 DualWeightAttention 中的 attn_weights = QK^T + mask
            # 我们希望特定的 query token 看到特定的 context (key)
            gate_mask = gate_bias.unsqueeze(1) # [batch, 1, 1, seq_len]
            gate_mask = gate_mask * attention_weight.unsqueeze(1) # [batch, 1, seq_len, seq_len]
            
        elif self.gate_type == "multiplicative":
            # 乘法门控同理
            gate_mask = torch.sigmoid(torch.matmul(mem_proj, key.transpose(-2, -1))).unsqueeze(1)
            gate_mask = gate_mask * attention_weight.unsqueeze(1)
        
        return gate_mask
    
    def sort_by_temporal(
        self,
        memories: List[dict],
        current_timestamp: int,
        topk: int = 2
    ) -> List[dict]:
        """
        按时序关系排序记忆
        
        Args:
            memories: 记忆列表
            current_timestamp: 当前时间戳
            topk: 返回数量
        
        Returns:
            sorted_memories: 按时序排序的记忆列表
        """
        if not memories:
            return []
        
        # 计算与当前时间的时间差
        def temporal_distance(mem):
            return abs(mem.get('timestamp', current_timestamp) - current_timestamp)
        
        # 按时间差排序 (优先选择最近的记忆)
        sorted_memories = sorted(memories, key=temporal_distance)
        
        return sorted_memories[:topk]
