"""
窄带宽注意力模块 - 类人脑稀疏注意力实现

核心原理:
1. 海马体记忆锚点 → 替代全局 KV-cache
2. 稀疏注意力掩码 → 只关注相关记忆
3. O(1) 复杂度 → 记忆锚点数量固定

对应大脑机制:
- 海马体 CA3: 模式补全（记忆召回）
- 海马体 CA1: 注意力门控（稀疏掩码）
- 前额叶: 工作记忆（窄窗口）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple


class MemoryAnchor:
    """记忆锚点 - 对应海马体的情景记忆节点"""
    
    def __init__(
        self,
        anchor_id: str,
        key: torch.Tensor,      # [num_heads, head_dim]
        value: torch.Tensor,    # [num_heads, head_dim]
        strength: float = 1.0,  # 记忆强度
        semantic: str = "",     # 语义描述
        timestamp: int = 0      # 创建时间
    ):
        self.anchor_id = anchor_id
        self.key = key
        self.value = value
        self.strength = strength
        self.semantic = semantic
        self.timestamp = timestamp
    
    def to_dict(self) -> dict:
        return {
            'anchor_id': self.anchor_id,
            'key': self.key.detach().cpu().numpy().tolist() if self.key is not None else None,
            'value': self.value.detach().cpu().numpy().tolist() if self.value is not None else None,
            'strength': self.strength,
            'semantic': self.semantic,
            'timestamp': self.timestamp
        }


class NarrowBandAttention(nn.Module):
    """
    窄带宽注意力层
    
    实现类人脑的稀疏注意力机制:
    1. 只关注记忆锚点 + 当前 token（工作记忆限制）
    2. 忽略其他历史 token（稀疏激活）
    3. 复杂度从 O(n) 降为 O(k)，k 是记忆锚点数量
    
    示例:
        标准注意力: 生成第 100 个 token，需计算与 99 个历史 token 的注意力
        窄带宽注意力: 只计算与 3-5 个记忆锚点的注意力
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        max_anchors: int = 5,      # 最大记忆锚点数（工作记忆容量）
        anchor_threshold: float = 0.3,  # 记忆强度阈值
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_anchors = max_anchors
        self.anchor_threshold = anchor_threshold
        
        # Q, K, V 投影（由外部 DualWeightLinear 处理，这里只做注意力计算）
        self.dropout = nn.Dropout(dropout)
        
        # 记忆锚点存储（对应海马体）
        self.anchor_store: List[MemoryAnchor] = []
        
    def forward(
        self,
        query: torch.Tensor,           # [batch, num_heads, 1, head_dim] - 当前 token
        key: torch.Tensor,             # [batch, num_heads, seq_len, head_dim] - 历史 KV
        value: torch.Tensor,           # [batch, num_heads, seq_len, head_dim]
        memory_anchors: Optional[List[Dict]] = None,  # 海马体召回的记忆锚点
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        窄带宽注意力前向传播
        
        Args:
            query: 当前 token 的 query 向量
            key: 历史 key 向量（如果使用窄带宽，这个会被忽略）
            value: 历史 value 向量
            memory_anchors: 海马体召回的记忆锚点列表
            attention_mask: 标准注意力掩码
        
        Returns:
            attn_output: 注意力输出
            attn_weights: 注意力权重（用于调试）
        """
        batch_size, num_heads, seq_len, _ = key.shape
        
        # ========== 类人脑稀疏注意力核心 ==========
        if memory_anchors and len(memory_anchors) > 0:
            # 使用记忆锚点构建稀疏 K, V
            sparse_key, sparse_value = self._build_sparse_kv(
                memory_anchors, 
                key[:, :, -1:, :],  # 当前 token 的 K
                value[:, :, -1:, :],  # 当前 token 的 V
                device=query.device
            )
            # 复杂度: O(k)，k = max_anchors (固定常数)
        else:
            # 回退到标准注意力（或滑动窗口）
            # 如果序列太长，使用滑动窗口
            if seq_len > self.max_anchors * 2:
                sparse_key, sparse_value = self._build_window_kv(key, value)
            else:
                sparse_key, sparse_value = key, value
        
        # ========== 标准注意力计算（但在稀疏 K, V 上）==========
        # Q * K^T / sqrt(d)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query, sparse_key.transpose(-1, -2)) * scale
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # * V
        attn_output = torch.matmul(attn_weights, sparse_value)
        
        return attn_output, attn_weights
    
    def _build_sparse_kv(
        self,
        memory_anchors: List[Dict],
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从记忆锚点构建稀疏 K, V
        
        对应大脑机制:
        - 海马体 CA3: 模式补全，召回相关记忆
        - 海马体 CA1: 注意力门控，筛选记忆锚点
        - 前额叶: 工作记忆容量限制
        """
        keys = []
        values = []
        
        # 1. 添加记忆锚点（类人脑：联想记忆）
        for anchor in memory_anchors[:self.max_anchors]:
            # 检查记忆强度（类人脑：记忆衰减）
            strength = anchor.get('activation_strength', 0.5)
            if strength < self.anchor_threshold:
                continue
            
            # 提取 K, V 特征
            if 'key_features' in anchor and anchor['key_features'] is not None:
                anchor_key = anchor['key_features']
                if not isinstance(anchor_key, torch.Tensor):
                    anchor_key = torch.tensor(anchor_key, device=device)
                keys.append(anchor_key)
            
            if 'value_features' in anchor and anchor['value_features'] is not None:
                anchor_value = anchor['value_features']
                if not isinstance(anchor_value, torch.Tensor):
                    anchor_value = torch.tensor(anchor_value, device=device)
                values.append(anchor_value)
        
        # 2. 添加当前 token（类人脑：感知输入）
        keys.append(current_key.squeeze(2))   # [num_heads, head_dim]
        values.append(current_value.squeeze(2))
        
        # 3. 堆叠为张量
        # [num_heads, num_anchors+1, head_dim]
        sparse_key = torch.stack(keys, dim=1).unsqueeze(0)   # [1, num_heads, num_anchors+1, head_dim]
        sparse_value = torch.stack(values, dim=1).unsqueeze(0)
        
        # 扩展 batch 维度
        sparse_key = sparse_key.transpose(1, 2)  # [1, num_anchors+1, num_heads, head_dim]
        sparse_key = sparse_key.transpose(2, 3)  # [1, num_heads, head_dim, num_anchors+1]
        sparse_key = sparse_key.transpose(2, 3)  # [1, num_heads, num_anchors+1, head_dim]
        
        sparse_value = sparse_value.transpose(1, 2)
        sparse_value = sparse_value.transpose(2, 3)
        sparse_value = sparse_value.transpose(2, 3)
        
        return sparse_key, sparse_value
    
    def _build_window_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        滑动窗口回退方案
        
        类人脑: 注意力的聚光灯效应，关注最近的上下文
        """
        seq_len = key.shape[2]
        window_size = self.max_anchors * 2
        
        # 只取最近的 window_size 个 token
        start = max(0, seq_len - window_size)
        window_key = key[:, :, start:, :]
        window_value = value[:, :, start:, :]
        
        return window_key, window_value


class NarrowBandAttentionWrapper(nn.Module):
    """
    窄带宽注意力包装器
    
    将 Qwen 的原始注意力层包装为窄带宽注意力
    保持与原始模型的兼容性
    """
    
    def __init__(
        self,
        original_attn: nn.Module,
        hidden_size: int = 1024,
        num_heads: int = 16,
        max_anchors: int = 5
    ):
        super().__init__()
        self.original_attn = original_attn
        self.narrow_band = NarrowBandAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            max_anchors=max_anchors
        )
        
        # 从原始注意力层提取参数
        self.q_proj = original_attn.q_proj if hasattr(original_attn, 'q_proj') else None
        self.k_proj = original_attn.k_proj if hasattr(original_attn, 'k_proj') else None
        self.v_proj = original_attn.v_proj if hasattr(original_attn, 'v_proj') else None
        self.o_proj = original_attn.o_proj if hasattr(original_attn, 'o_proj') else None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_anchors: Optional[List[Dict]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播: 优先使用窄带宽注意力，回退到原始注意力
        """
        # 如果有记忆锚点，使用窄带宽注意力
        if memory_anchors and len(memory_anchors) > 0:
            return self._narrow_band_forward(hidden_states, memory_anchors, **kwargs)
        else:
            # 回退到原始注意力
            return self.original_attn(hidden_states, **kwargs)
    
    def _narrow_band_forward(
        self,
        hidden_states: torch.Tensor,
        memory_anchors: List[Dict],
        **kwargs
    ) -> torch.Tensor:
        """窄带宽注意力前向传播"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. 投影 Q, K, V
        query = self.q_proj(hidden_states[:, -1:, :])  # 只处理最后一个 token
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 2. 重塑为多头格式
        query = query.view(batch_size, 1, self.narrow_band.num_heads, self.narrow_band.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.narrow_band.num_heads, self.narrow_band.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.narrow_band.num_heads, self.narrow_band.head_dim).transpose(1, 2)
        
        # 3. 窄带宽注意力计算
        attn_output, _ = self.narrow_band(query, key, value, memory_anchors, **kwargs)
        
        # 4. 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.narrow_band.hidden_size)
        
        # 5. 输出投影
        output = self.o_proj(attn_output)
        
        return output


# ==================== 集成工具函数 ====================

def integrate_narrow_band_attention(model, max_anchors: int = 5):
    """
    将窄带宽注意力集成到 Qwen 模型中
    
    Args:
        model: QwenModelWrapper 实例
        max_anchors: 最大记忆锚点数
    """
    from core.dual_weight_layers import DualWeightLinear
    
    print("\n[集成] 开始集成窄带宽注意力...")
    
    replaced_count = 0
    
    # 遍历所有 Transformer 层
    for name, module in model.base_model.named_modules():
        # 寻找注意力层
        if hasattr(module, 'self_attn') or hasattr(module, 'attn'):
            attn = getattr(module, 'self_attn', None) or getattr(module, 'attn', None)
            
            if attn is not None and not isinstance(attn, NarrowBandAttentionWrapper):
                try:
                    # 包装为窄带宽注意力
                    wrapper = NarrowBandAttentionWrapper(
                        original_attn=attn,
                        hidden_size=model.base_model.config.hidden_size,
                        num_heads=model.base_model.config.num_attention_heads,
                        max_anchors=max_anchors
                    )
                    
                    # 替换
                    if hasattr(module, 'self_attn'):
                        setattr(module, 'self_attn', wrapper)
                    else:
                        setattr(module, 'attn', wrapper)
                    
                    replaced_count += 1
                except Exception as e:
                    print(f"  [!] 替换注意力层失败 {name}: {e}")
    
    print(f"[OK] 已集成 {replaced_count} 个窄带宽注意力层")
    print(f"  - 最大记忆锚点数: {max_anchors}")
    print(f"  - 注意力复杂度: O({max_anchors}) (类人脑稀疏激活)")


def extract_memory_anchors_from_kv(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    important_positions: List[int],
    semantic_pointers: List[str]
) -> List[Dict]:
    """
    从 KV-cache 提取记忆锚点
    
    对应大脑机制: 海马体将工作记忆编码为长期记忆
    
    Args:
        past_key_values: KV-cache
        important_positions: 重要位置索引
        semantic_pointers: 语义指针
    
    Returns:
        memory_anchors: 记忆锚点列表
    """
    memory_anchors = []
    
    for i, pos in enumerate(important_positions):
        if i >= len(semantic_pointers):
            break
        
        # 从所有层提取 K, V 特征
        keys = []
        values = []
        
        for layer_kv in past_key_values:
            key, value = layer_kv
            # key/value: [batch, num_heads, seq_len, head_dim]
            keys.append(key[:, :, pos, :].mean(dim=1))   # 平均所有头
            values.append(value[:, :, pos, :].mean(dim=1))
        
        # 平均所有层
        anchor_key = torch.stack(keys).mean(dim=0)   # [num_heads, head_dim]
        anchor_value = torch.stack(values).mean(dim=0)
        
        memory_anchors.append({
            'anchor_id': f'anchor_{pos}',
            'key_features': anchor_key,
            'value_features': anchor_value,
            'activation_strength': 1.0,
            'semantic': semantic_pointers[i]
        })
    
    return memory_anchors
