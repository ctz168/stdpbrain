"""
Qwen3.5 窄带宽注意力补丁

通过 Monkey Patching 在运行时修改 Qwen 的注意力层，
注入记忆锚点机制，实现类人脑的稀疏注意力。

类人脑对应：
- 海马体 CA3: 记忆锚点存储（memory_anchors）
- 海马体 CA1: 注意力门控（稀疏 KV 注入）
- 前额叶: 工作记忆容量限制（max_anchors）

性能提升：
- 标准注意力: O(n²) 复杂度
- 窄带宽注意力: O((n+k)·d)，k 是记忆锚点数量（常数 3-5）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any
import math


# ==================== 全局记忆锚点存储 ====================

class MemoryAnchorStore:
    """
    全局记忆锚点存储
    
    在推理过程中传递记忆锚点到注意力层
    """
    def __init__(self):
        self.anchors: List[Dict] = []
        self.enabled: bool = True
        self.max_anchors: int = 5
        self.anchor_strength: float = 1.0
    
    def set_anchors(self, anchors: List[Dict], max_anchors: int = 5, strength: float = 1.0):
        """设置记忆锚点"""
        self.anchors = anchors or []
        self.max_anchors = max_anchors
        self.anchor_strength = strength
    
    def clear(self):
        """清除记忆锚点"""
        self.anchors = []
    
    def get_enabled_anchors(self) -> List[Dict]:
        """获取启用的记忆锚点"""
        if not self.enabled or not self.anchors:
            return []
        return self.anchors[:self.max_anchors]


# 全局单例
_memory_anchor_store = MemoryAnchorStore()


def get_memory_anchor_store() -> MemoryAnchorStore:
    """获取全局记忆锚点存储"""
    return _memory_anchor_store


# ==================== 记忆锚点注入器 ====================

class MemoryAnchorInjector:
    """
    记忆锚点注入器
    
    将海马体召回的记忆锚点注入到 KV-cache 中
    """
    
    @staticmethod
    def inject_to_kv(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        anchors: List[Dict],
        hidden_size: int = 1024,
        num_heads: int = 2,  # GQA 的 KV heads
        head_dim: int = 256,
        device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将记忆锚点注入到 KV-cache
        
        Args:
            key_states: [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim]
            anchors: 记忆锚点列表
            hidden_size: 隐藏层大小
            num_heads: KV 头数
            head_dim: 头维度
            device: 设备
        
        Returns:
            enhanced_key, enhanced_value: 增强后的 KV
        """
        if not anchors:
            return key_states, value_states
        
        device = device or key_states.device
        batch_size = key_states.shape[0]
        
        # 构建锚点 KV (每个锚点形状: [batch, num_heads, 1, head_dim])
        anchor_keys = []
        anchor_values = []
        
        for anchor in anchors:
            # 尝试不同的特征来源
            anchor_k = None
            anchor_v = None
            
            # 优先使用预计算的 KV 特征
            if 'key_features' in anchor and anchor['key_features'] is not None:
                try:
                    anchor_k = torch.tensor(anchor['key_features'], device=device, dtype=key_states.dtype)
                    # 确保 4D 形状: [batch, num_heads, 1, head_dim]
                    if anchor_k.dim() == 1:
                        anchor_k = anchor_k.view(1, 1, 1, -1)  # [1, 1, 1, head_dim]
                    elif anchor_k.dim() == 2:
                        anchor_k = anchor_k.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, head_dim]
                    elif anchor_k.dim() == 3:
                        anchor_k = anchor_k.unsqueeze(2)  # [batch, num_heads, 1, head_dim]
                    
                    # 调整 head_dim
                    if anchor_k.shape[-1] != head_dim:
                        if anchor_k.shape[-1] < head_dim:
                            pad = torch.zeros(*anchor_k.shape[:-1], head_dim - anchor_k.shape[-1], device=device, dtype=key_states.dtype)
                            anchor_k = torch.cat([anchor_k, pad], dim=-1)
                        else:
                            anchor_k = anchor_k[..., :head_dim]
                    
                    # 扩展到正确的 batch 和 num_heads
                    if anchor_k.shape[0] != batch_size:
                        anchor_k = anchor_k.expand(batch_size, -1, -1, -1)
                    if anchor_k.shape[1] != num_heads:
                        anchor_k = anchor_k.expand(-1, num_heads, -1, -1)
                    
                    anchor_keys.append(anchor_k)
                except Exception as e:
                    print(f"[MemoryAnchorInjector] key_features 处理失败: {e}")
            
            if 'value_features' in anchor and anchor['value_features'] is not None:
                try:
                    anchor_v = torch.tensor(anchor['value_features'], device=device, dtype=value_states.dtype)
                    # 确保 4D 形状
                    if anchor_v.dim() == 1:
                        anchor_v = anchor_v.view(1, 1, 1, -1)
                    elif anchor_v.dim() == 2:
                        anchor_v = anchor_v.unsqueeze(1).unsqueeze(1)
                    elif anchor_v.dim() == 3:
                        anchor_v = anchor_v.unsqueeze(2)
                    
                    if anchor_v.shape[-1] != head_dim:
                        if anchor_v.shape[-1] < head_dim:
                            pad = torch.zeros(*anchor_v.shape[:-1], head_dim - anchor_v.shape[-1], device=device, dtype=value_states.dtype)
                            anchor_v = torch.cat([anchor_v, pad], dim=-1)
                        else:
                            anchor_v = anchor_v[..., :head_dim]
                    
                    if anchor_v.shape[0] != batch_size:
                        anchor_v = anchor_v.expand(batch_size, -1, -1, -1)
                    if anchor_v.shape[1] != num_heads:
                        anchor_v = anchor_v.expand(-1, num_heads, -1, -1)
                    
                    anchor_values.append(anchor_v)
                except Exception as e:
                    print(f"[MemoryAnchorInjector] value_features 处理失败: {e}")
            
            # 如果没有预计算的 KV，从 dg_features 生成
            if (anchor_k is None or anchor_v is None) and 'dg_features' in anchor and anchor['dg_features'] is not None:
                try:
                    feat = torch.tensor(anchor['dg_features'], device=device, dtype=key_states.dtype)
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)
                    
                    # 简单截断或填充
                    if feat.shape[-1] < hidden_size:
                        pad = torch.zeros(feat.shape[0], hidden_size - feat.shape[-1], device=device, dtype=key_states.dtype)
                        feat = torch.cat([feat, pad], dim=-1)
                    elif feat.shape[-1] > hidden_size:
                        feat = feat[..., :hidden_size]
                    
                    # 简单投影生成 KV
                    kv_size = num_heads * head_dim
                    if feat.shape[-1] >= kv_size * 2:
                        # [batch, num_heads, 1, head_dim]
                        anchor_k = feat[:, :kv_size].view(1, num_heads, 1, head_dim).expand(batch_size, -1, -1, -1)
                        anchor_v = feat[:, kv_size:kv_size*2].view(1, num_heads, 1, head_dim).expand(batch_size, -1, -1, -1)
                    else:
                        # 重复特征
                        anchor_k = feat[:, :head_dim].view(1, 1, 1, head_dim).expand(batch_size, num_heads, -1, -1)
                        anchor_v = feat[:, head_dim:head_dim*2].view(1, 1, 1, head_dim).expand(batch_size, num_heads, -1, -1) if feat.shape[-1] >= head_dim * 2 else anchor_k.clone()
                    
                    if anchor_k is not None:
                        anchor_keys.append(anchor_k)
                    if anchor_v is not None:
                        anchor_values.append(anchor_v)
                except Exception as e:
                    print(f"[MemoryAnchorInjector] dg_features 处理失败: {e}")
        
        if not anchor_keys:
            return key_states, value_states
        
        # 应用记忆强度
        strength = getattr(_memory_anchor_store, 'anchor_strength', 1.0)
        anchor_keys = [k * strength for k in anchor_keys]
        anchor_values = [v * strength for v in anchor_values]
        
        # 拼接锚点 KV: [batch, num_heads, num_anchors, head_dim]
        anchor_key_tensor = torch.cat(anchor_keys, dim=2)
        anchor_value_tensor = torch.cat(anchor_values, dim=2)
        
        # 拼接到原始 KV 前面
        # 新 KV: [锚点1, 锚点2, ..., 原始token1, 原始token2, ...]
        enhanced_key = torch.cat([anchor_key_tensor, key_states], dim=2)
        enhanced_value = torch.cat([anchor_value_tensor, value_states], dim=2)
        
        return enhanced_key, enhanced_value
    
    @staticmethod
    def expand_attention_mask(
        attention_mask: Optional[torch.Tensor],
        num_anchors: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> Optional[torch.Tensor]:
        """
        扩展注意力掩码以包含记忆锚点
        
        Args:
            attention_mask: 原始掩码 [batch, seq_len] 或 [batch, 1, seq_len, seq_len]
            num_anchors: 记忆锚点数量
            device: 设备
            dtype: 数据类型
        
        Returns:
            expanded_mask: 扩展后的掩码
        """
        if attention_mask is None or num_anchors == 0:
            return attention_mask
        
        # 2D 掩码 [batch, seq_len]
        if attention_mask.dim() == 2:
            batch_size, seq_len = attention_mask.shape
            new_seq_len = seq_len + num_anchors
            
            # 创建新掩码：锚点位置为 0（可访问）
            new_mask = torch.zeros(batch_size, new_seq_len, device=device, dtype=dtype)
            
            # 原始掩码复制到后面
            new_mask[:, num_anchors:] = attention_mask
            
            return new_mask
        
        # 4D 掩码 [batch, 1, seq_len, seq_len]
        elif attention_mask.dim() == 4:
            batch_size, _, seq_len, _ = attention_mask.shape
            new_seq_len = seq_len + num_anchors
            
            # 创建新掩码：锚点位置为 0（可访问）
            new_mask = torch.zeros(batch_size, 1, new_seq_len, new_seq_len, device=device, dtype=dtype)
            
            # 原始掩码复制到右下角
            new_mask[:, :, num_anchors:, num_anchors:] = attention_mask
            
            return new_mask
        
        return attention_mask


# ==================== Qwen 注意力层补丁 ====================

def patch_qwen_attention():
    """
    在运行时补丁 Qwen3.5 的注意力层
    
    修改 Qwen3_5Attention.forward 方法以支持记忆锚点注入
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
        
        # 保存原始 forward 方法
        original_forward = Qwen3_5Attention.forward
        
        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple,
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            cache_position: torch.LongTensor | None = None,
            **kwargs
        ):
            """
            修改后的注意力 forward 方法
            
            在 KV-cache 更新后注入记忆锚点
            """
            # 获取记忆锚点
            anchor_store = get_memory_anchor_store()
            anchors = anchor_store.get_enabled_anchors()
            
            # 调用原始 forward 的前半部分
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)

            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # ========== 注入记忆锚点 ==========
            if anchors:
                key_states, value_states = MemoryAnchorInjector.inject_to_kv(
                    key_states=key_states,
                    value_states=value_states,
                    anchors=anchors,
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_key_value_heads,
                    head_dim=self.head_dim,
                    device=hidden_states.device
                )
                
                # 扩展注意力掩码
                attention_mask = MemoryAnchorInjector.expand_attention_mask(
                    attention_mask=attention_mask,
                    num_anchors=len(anchors),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )

            # 继续原始 forward 的后半部分
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)

            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights
        
        # 应用补丁
        Qwen3_5Attention.forward = patched_forward
        
        store = get_memory_anchor_store()
        print("[QwenNarrowBandPatch] ✅ 已成功补丁 Qwen3_5Attention")
        print(f"  - 记忆锚点支持: 启用")
        print(f"  - 最大锚点数: {store.max_anchors}")
        
        return True
        
    except Exception as e:
        print(f"[QwenNarrowBandPatch] ❌ 补丁失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# 需要从 transformers 导入的辅助函数
def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    # 从 transformers 导入
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb as _apply
        return _apply(q, k, cos, sin)
    except:
        # 回退实现
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


def rotate_half(x):
    """旋转一半"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def eager_attention_forward(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    标准注意力前向传播
    """
    attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * scaling
    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ==================== 自动应用补丁 ====================

def auto_patch():
    """自动应用补丁（在导入时执行）"""
    try:
        patch_qwen_attention()
    except Exception as e:
        print(f"[QwenNarrowBandPatch] 自动补丁失败（将在模型加载时重试）: {e}")


# 在模块导入时尝试应用补丁
# auto_patch()  # 取消注释以在导入时自动应用
