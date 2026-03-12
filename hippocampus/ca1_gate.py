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
        hidden_size: int = 1024,       # 模型 hidden size
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
        else:
            # 简化处理：随机特征
            mem_features = torch.randn(self.feature_dim, device=query.device)
        
        # ========== 时序编码 (可选) ==========
        if self.temporal_encoding and 'timestamp' in memory_data:
            timestamp = torch.tensor(
                [[memory_data['timestamp']]], 
                dtype=torch.float32, 
                device=query.device
            )
            temporal_feature = self.temporal_encoder(timestamp)
            mem_features = mem_features + temporal_feature.squeeze(0)
        
        # ========== 计算注意力权重 ==========
        # 将记忆特征与 query/key 结合
        query_mean = query.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        
        combined = torch.cat([
            mem_features.unsqueeze(0).expand(batch_size, -1),
            query_mean.squeeze(1)
        ], dim=-1)
        
        attention_score = self.attention_scorer(combined)  # [batch, 1]
        attention_weight = torch.sigmoid(attention_score)
        
        # ========== 生成门控掩码 ==========
        if self.gate_type == "additive":
            # 加法门控：生成偏置项加到注意力分数上
            gate_projection = self.gate_projection(mem_features.unsqueeze(0))
            gate_mask = gate_projection.view(batch_size, 1, 1, hidden_size)
            gate_mask = gate_mask.expand(-1, -1, seq_len, -1)
            
            # 与 key 做外积得到注意力偏置
            gate_mask = torch.matmul(gate_mask, key.unsqueeze(1).transpose(-2, -1))
            gate_mask = gate_mask * attention_weight.view(batch_size, 1, 1, 1)
            
        elif self.gate_type == "multiplicative":
            # 乘法门控：直接缩放注意力
            gate_projection = self.gate_projection(mem_features.unsqueeze(0))
            gate_mask = gate_projection.view(batch_size, 1, 1, hidden_size)
            gate_mask = gate_mask.expand(-1, -1, seq_len, -1)
            
            # 与 key 逐元素相乘
            gate_mask = gate_mask * key.unsqueeze(1)
            gate_mask = torch.matmul(gate_mask, key.unsqueeze(1).transpose(-2, -1))
        
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
    
    def encode_temporal_context(
        self,
        memory_chain: List[dict]
    ) -> str:
        """
        编码时序上下文
        
        Args:
            memory_chain: 记忆链条
        
        Returns:
            temporal_context: 时序上下文字符串
        """
        if not memory_chain:
            return ""
        
        # 按时间戳排序
        sorted_chain = sorted(memory_chain, key=lambda m: m.get('timestamp', 0))
        
        # 生成时序骨架
        temporal_skeleton = " -> ".join([
            f"{m.get('semantic_pointer', '?')}@{m.get('timestamp', 0)}"
            for m in sorted_chain
        ])
        
        return temporal_skeleton


class TemporalEncoder(nn.Module):
    """
    时序编码器
    
    为记忆绑定精确的时间戳信息
    """
    
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 时间戳嵌入
        self.timestamp_embedding = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim)
        )
        
        # 位置编码 (类似 Transformer)
        self.position_encoding = self._generate_position_encoding(feature_dim)
    
    def _generate_position_encoding(self, dim: int, max_len: int = 10000) -> torch.Tensor:
        """生成正弦位置编码"""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(
        self,
        features: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        时序编码
        
        Args:
            features: 输入特征 [batch, seq, dim]
            timestamps: 时间戳 [batch, seq, 1]
        
        Returns:
            encoded: 时序编码后的特征
        """
        # 时间戳编码
        time_code = self.timestamp_embedding(timestamps)
        
        # 位置编码
        seq_len = features.shape[1]
        pos_code = self.position_encoding[:, :seq_len, :]
        
        # 融合
        encoded = features + time_code + pos_code
        
        return encoded
