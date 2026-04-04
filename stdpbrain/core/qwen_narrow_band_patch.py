"""
Qwen3.5 True Sparse Narrow-Band Attention Patch

Core Improvement:
- No longer concatenate anchors (increases computation)
- Implement true sparse attention: keep only recent window + memory anchors
- Complexity reduced from O(n^2) to O(n * (W+K))

Human Brain Analogy:
- Working memory capacity limit: W=7 (7 +/- 2 rule)
- Hippocampal memory anchors: K=3-5
- Attention sparsity: focus only on key information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any
import math
import logging


logger = logging.getLogger(__name__)


# ==================== Global Memory Anchor Store ====================

class MemoryAnchorStore:
    """Global memory anchor storage"""
    def __init__(self):
        self.anchors: List[Dict] = []
        self.enabled: bool = True  # 启用稀疏注意力压缩（模拟人脑注意力机制）
        self.max_anchors: int = 5
        self.anchor_strength: float = 1.0
        self.window_size: int = 256  # 窗口大小（增大到256，保持更多上下文）
    
    def set_anchors(self, anchors: List[Dict], max_anchors: int = 5, strength: float = 1.0):
        """Set memory anchors"""
        self.anchors = anchors or []
        self.max_anchors = max_anchors
        self.anchor_strength = strength
    
    def clear(self):
        """Clear memory anchors"""
        self.anchors = []
    
    def get_enabled_anchors(self) -> List[Dict]:
        """Get enabled memory anchors"""
        if not self.enabled or not self.anchors:
            return []
        return self.anchors[:self.max_anchors]


# Global singleton
_memory_anchor_store = MemoryAnchorStore()


def get_memory_anchor_store() -> MemoryAnchorStore:
    """Get global memory anchor storage"""
    return _memory_anchor_store


# ==================== True Sparse Attention Implementation ====================

class SparseAttentionCompressor:
    """
    Sparse Attention Compressor
    
    Compresses KV to: recent W tokens + K memory anchors
    Implements human-like narrow-band attention
    
    Key Fix: Properly handle RoPE when concatenating anchors and window
    
    CPU优化: 缓存attention mask避免每步重新分配
    """
    
    # 类级别缓存（避免每层实例重复分配）
    _mask_cache: Optional[torch.Tensor] = None
    _mask_cache_key: Optional[tuple] = None
    
    @staticmethod
    def reapply_rope(
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_offset: int = 0
    ) -> torch.Tensor:
        """
        重新应用RoPE位置编码
        
        Args:
            key_states: [batch, num_heads, seq_len, head_dim]
            cos, sin: RoPE的cos和sin [seq_len, head_dim]
            position_offset: 位置偏移量
        
        Returns:
            重新应用RoPE后的key_states
        """
        # 获取序列长度
        seq_len = key_states.shape[2]
        
        # 选择对应位置的cos和sin
        # cos, sin形状: [seq_len, head_dim] 或 [1, seq_len, head_dim]
        if cos.dim() == 3:
            cos_pos = cos[:, position_offset:position_offset+seq_len, :]
            sin_pos = sin[:, position_offset:position_offset+seq_len, :]
        else:
            cos_pos = cos[position_offset:position_offset+seq_len, :].unsqueeze(0)
            sin_pos = sin[position_offset:position_offset+seq_len, :].unsqueeze(0)
        
        # 扩展到正确的维度 [batch, num_heads, seq_len, head_dim]
        # cos_pos: [1, seq_len, head_dim] -> [batch, num_heads, seq_len, head_dim]
        cos_pos = cos_pos.unsqueeze(0).expand(key_states.shape[0], -1, -1, -1)
        cos_pos = cos_pos.expand(-1, key_states.shape[1], -1, -1)
        
        sin_pos = sin_pos.unsqueeze(0).expand(key_states.shape[0], -1, -1, -1)
        sin_pos = sin_pos.expand(-1, key_states.shape[1], -1, -1)
        
        # 应用旋转位置编码（RoPE）
        # rotate_half: 将向量分成两半并旋转
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        # RoPE公式: k' = k * cos + rotate_half(k) * sin
        key_rotated = key_states * cos_pos + rotate_half(key_states) * sin_pos
        
        return key_rotated
    
    @staticmethod
    def apply_rope_with_positions(
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        对指定位置索引应用RoPE（仅用于新构造的anchor keys）。

        Args:
            key_states: [batch, num_heads, seq_len, head_dim]
            cos, sin: RoPE缓存
            positions: [seq_len]，每个token对应的位置索引
        """
        if key_states.shape[2] == 0:
            return key_states

        # 统一取出 [seq_len, head_dim]
        if cos.dim() == 3:
            base_cos = cos[0]
            base_sin = sin[0]
        else:
            base_cos = cos
            base_sin = sin

        max_pos = base_cos.shape[0] - 1
        positions = positions.clamp(min=0, max=max_pos).to(torch.long).to(key_states.device)

        cos_pos = base_cos.index_select(0, positions).unsqueeze(0).unsqueeze(0)
        sin_pos = base_sin.index_select(0, positions).unsqueeze(0).unsqueeze(0)
        cos_pos = cos_pos.expand(key_states.shape[0], key_states.shape[1], -1, -1)
        sin_pos = sin_pos.expand(key_states.shape[0], key_states.shape[1], -1, -1)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        return key_states * cos_pos + rotate_half(key_states) * sin_pos

    @staticmethod
    def compress_kv(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        anchors: List[Dict],
        window_size: int = 64,
        num_heads: int = 2,
        head_dim: int = 256,
        device: torch.device = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV to sparse representation
        
        Args:
            key_states: [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim]
            anchors: Memory anchor list
            window_size: Window size (how many recent tokens to keep)
            num_heads: Number of KV heads
            head_dim: Head dimension
            device: Device
        
        Returns:
            compressed_key, compressed_value: [batch, num_heads, window_size+num_anchors, head_dim]
        """
        device = device or key_states.device
        batch_size = key_states.shape[0]
        seq_len = key_states.shape[2]
        
        # ========== 1. Extract recent window ==========
        if seq_len <= window_size:
            # Sequence is short, return as-is
            window_keys = key_states
            window_values = value_states
        else:
            # Keep the most recent window_size tokens
            window_keys = key_states[:, :, -window_size:, :]
            window_values = value_states[:, :, -window_size:, :]
        
        # ========== 2. Build anchor KV ==========
        anchor_keys_list = []
        anchor_values_list = []
        
        for anchor in anchors[:5]:  # At most 5 anchors
            # Extract KV from anchor features
            anchor_k = None
            anchor_v = None
            
            # Try to extract from key_features
            if 'key_features' in anchor and anchor['key_features'] is not None:
                feat = torch.tensor(anchor['key_features'], device=device, dtype=key_states.dtype)
                if feat.dim() == 1:
                    feat = feat.view(1, 1, 1, -1)
                elif feat.dim() == 2:
                    feat = feat.unsqueeze(0).unsqueeze(0)
                elif feat.dim() == 3:
                    feat = feat.unsqueeze(0)
                
                # Adjust to correct head_dim
                if feat.shape[-1] != head_dim:
                    if feat.shape[-1] < head_dim:
                        pad = torch.zeros(*feat.shape[:-1], head_dim - feat.shape[-1], device=device, dtype=key_states.dtype)
                        feat = torch.cat([feat, pad], dim=-1)
                    else:
                        feat = feat[..., :head_dim]
                
                # Expand to correct batch and num_heads
                if feat.shape[0] != batch_size:
                    feat = feat.expand(batch_size, -1, -1, -1)
                if feat.shape[1] != num_heads:
                    feat = feat.expand(-1, num_heads, -1, -1)
                
                anchor_k = feat
            
            # Try to extract from value_features
            if 'value_features' in anchor and anchor['value_features'] is not None:
                feat = torch.tensor(anchor['value_features'], device=device, dtype=value_states.dtype)
                if feat.dim() == 1:
                    feat = feat.view(1, 1, 1, -1)
                elif feat.dim() == 2:
                    feat = feat.unsqueeze(0).unsqueeze(0)
                elif feat.dim() == 3:
                    feat = feat.unsqueeze(0)
                
                if feat.shape[-1] != head_dim:
                    if feat.shape[-1] < head_dim:
                        pad = torch.zeros(*feat.shape[:-1], head_dim - feat.shape[-1], device=device, dtype=value_states.dtype)
                        feat = torch.cat([feat, pad], dim=-1)
                    else:
                        feat = feat[..., :head_dim]
                
                if feat.shape[0] != batch_size:
                    feat = feat.expand(batch_size, -1, -1, -1)
                if feat.shape[1] != num_heads:
                    feat = feat.expand(-1, num_heads, -1, -1)
                
                anchor_v = feat
            
            # If no pre-computed features, try to generate from dg_features
            if (anchor_k is None or anchor_v is None) and 'dg_features' in anchor and anchor['dg_features'] is not None:
                feat = torch.tensor(anchor['dg_features'], device=device, dtype=key_states.dtype)
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                
                # Simple projection to generate K and V
                hidden_size = num_heads * head_dim
                if feat.shape[-1] < hidden_size:
                    pad = torch.zeros(feat.shape[0], hidden_size - feat.shape[-1], device=device, dtype=key_states.dtype)
                    feat = torch.cat([feat, pad], dim=-1)
                else:
                    feat = feat[..., :hidden_size]
                
                # Split into K and V
                kv_size = num_heads * head_dim
                anchor_k = feat[:, :kv_size].view(1, num_heads, 1, head_dim).expand(batch_size, -1, -1, -1)
                anchor_v = feat[:, kv_size:kv_size*2].view(1, num_heads, 1, head_dim).expand(batch_size, -1, -1, -1) if feat.shape[-1] >= kv_size * 2 else anchor_k.clone()
            
            # Add to lists
            if anchor_k is not None:
                anchor_keys_list.append(anchor_k)
            if anchor_v is not None:
                anchor_values_list.append(anchor_v)
        
        # ========== 3. Combine anchors and window ==========
        if anchor_keys_list:
            # Apply anchor strength
            strength = _memory_anchor_store.anchor_strength
            anchor_keys_tensor = torch.cat(anchor_keys_list, dim=2) * strength
            anchor_values_tensor = torch.cat(anchor_values_list, dim=2) * strength
            
            num_anchors = anchor_keys_tensor.shape[2]
            num_window = window_keys.shape[2]

            # 只对anchor应用RoPE，window部分保持原始RoPE编码，避免“二次RoPE”
            if cos is not None and sin is not None and num_anchors > 0:
                window_start_pos = max(0, seq_len - num_window)
                anchor_start_pos = max(0, window_start_pos - num_anchors)
                anchor_positions = torch.arange(
                    anchor_start_pos,
                    anchor_start_pos + num_anchors,
                    device=device
                )
                anchor_keys_tensor = SparseAttentionCompressor.apply_rope_with_positions(
                    anchor_keys_tensor, cos, sin, anchor_positions
                )
            
            # Concatenate: anchors + recent window
            compressed_keys = torch.cat([anchor_keys_tensor, window_keys], dim=2)
            compressed_values = torch.cat([anchor_values_tensor, window_values], dim=2)
            
            logger.debug(
                "[SparseAttn] KV拼接完成: %s anchors + %s window = %s total",
                num_anchors,
                num_window,
                compressed_keys.shape[2],
            )
        else:
            # No anchors, use only window
            compressed_keys = window_keys
            compressed_values = window_values
        
        return compressed_keys, compressed_values
    
    @staticmethod
    def adjust_attention_mask(
        attention_mask: Optional[torch.Tensor],
        num_anchors: int,
        window_size: int,
        original_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> Optional[torch.Tensor]:
        """
        Adjust attention mask to match compressed KV
        
        Compressed KV length = num_anchors + min(seq_len, window_size)
        Query length unchanged
        
        Key fix: Anchors should be visible to all queries (global memory)
                 Window should follow causal rules
        """
        if attention_mask is None:
            return attention_mask
        
        # Calculate compressed KV length
        compressed_kv_len = num_anchors + min(original_seq_len, window_size)
        query_len = attention_mask.shape[-2]  # Query length
        
        # 4D mask [batch, 1, query_len, kv_len]
        if attention_mask.dim() == 4:
            batch_size = attention_mask.shape[0]
            
            # ========== CPU优化: 使用缓存的mask ==========
            cache_key = (batch_size, query_len, num_anchors, window_size, original_seq_len)
            if (SparseAttentionCompressor._mask_cache is not None 
                and SparseAttentionCompressor._mask_cache_key == cache_key
                and SparseAttentionCompressor._mask_cache.shape == (batch_size, 1, query_len, compressed_kv_len)):
                return SparseAttentionCompressor._mask_cache
            
            # Create new mask: all positions initialized to 0 (can attend)
            new_mask = torch.zeros(batch_size, 1, query_len, compressed_kv_len, device=device, dtype=dtype)
            
            # ========== Anchors部分：对所有query可见（全局记忆）==========
            # 不需要mask，保持为0即可
            
            # ========== Window部分：遵循causal规则 ==========
            if original_seq_len <= window_size:
                # 序列短，直接复制causal mask
                new_mask[:, :, :, num_anchors:] = attention_mask
            else:
                # 序列长，需要调整causal mask
                # Query只能看到之前的token
                
                # 获取原始causal mask的最后window_size列
                # attention_mask shape: [batch, 1, query_len, original_seq_len]
                window_mask = attention_mask[:, :, :, -window_size:]  # [batch, 1, query_len, window_size]
                new_mask[:, :, :, num_anchors:] = window_mask
            
            # ========== CPU优化: 缓存mask ==========
            SparseAttentionCompressor._mask_cache = new_mask
            SparseAttentionCompressor._mask_cache_key = cache_key
            
            return new_mask
        
        # 2D mask [batch, seq_len]
        elif attention_mask.dim() == 2:
            batch_size, seq_len = attention_mask.shape
            # Create new mask (padding mask, not causal mask)
            new_mask = torch.zeros(batch_size, compressed_kv_len, device=device, dtype=dtype)
            
            # Anchors: all accessible (no mask)
            # Window: copy from original
            if original_seq_len <= window_size:
                new_mask[:, num_anchors:] = attention_mask
            else:
                new_mask[:, num_anchors:] = attention_mask[:, -window_size:]
            
            return new_mask
        
        return attention_mask


# ==================== Qwen Attention Layer Patch ====================


def patch_qwen_attention():
    """
    Patch Qwen attention layer at runtime (conditional on model architecture)
    
    Supports: Qwen3.5, Qwen2, and gracefully degrades for other models.
    Implements true narrow-band sparse attention:
    - KV compression: keep recent window + memory anchors
    - Complexity: O(n * (W + K)) instead of O(n^2)
    """
    AttentionClass = None
    attn_module_path = None
    
    # Try Qwen3.5 first, then Qwen2, then gracefully skip
    for cls_name, module_path in [
        ('Qwen3_5Attention', 'transformers.models.qwen3_5.modeling_qwen3_5'),
        ('Qwen2Attention', 'transformers.models.qwen2.modeling_qwen2'),
    ]:
        try:
            module = __import__(module_path, fromlist=[cls_name])
            AttentionClass = getattr(module, cls_name)
            attn_module_path = module_path
            print(f"[QwenNarrowBandPatch] Found {cls_name} from {module_path}")
            break
        except (ImportError, AttributeError):
            continue
    
    if AttentionClass is None:
        print("[QwenNarrowBandPatch] [SKIP] No compatible Qwen attention class found. Narrow-band patch disabled.")
        return False
    
    try:
        # Save original forward method
        original_forward = AttentionClass.forward
        
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
            Modified attention forward method
            
            Implements KV compression instead of KV expansion
            """
            # Get memory anchors
            anchor_store = get_memory_anchor_store()
            anchors = anchor_store.get_enabled_anchors()
            
            # First half of forward pass (compute Q, K, V)
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

            # ========== True sparse attention: KV compression ==========
            original_seq_len = key_states.shape[2]
            
            # 关键修复：仅在有记忆锚点时才压缩 KV
            # 无锚点时保持完整注意力，避免丢失上下文导致输出退化
            num_anchors = len(anchors) if anchors else 0
            if anchor_store.enabled and num_anchors > 0 and original_seq_len > anchor_store.window_size:
                
                # Compress KV: keep only window + anchors
                key_states, value_states = SparseAttentionCompressor.compress_kv(
                    key_states=key_states,
                    value_states=value_states,
                    anchors=anchors if anchors else [],  # 允许空锚点列表
                    window_size=anchor_store.window_size,
                    num_heads=self.config.num_key_value_heads,
                    head_dim=self.head_dim,
                    device=hidden_states.device,
                    cos=cos,  # 传递position embedding
                    sin=sin   # 传递position embedding
                )
                
                # ========== 关键：调整Q的position以匹配拼接后的KV ==========
                # Q应该attend到KV的最后位置
                # 拼接后的KV: [anchors...window...]
                # Q的position应该对应window的末尾
                
                # 方法：重新应用RoPE到Q，使用正确的position offset
                if num_anchors > 0 and cos is not None and sin is not None:
                    # Q的position需要偏移num_anchors
                    # 因为现在KV的position是 0, 1, 2, ..., num_anchors+window_size-1
                    # Q应该对应position num_anchors+window_size-1（或当前实际position）
                    
                    # 但实际上，Q的position已经在前面应用过了
                    # 我们需要"撤销"旧的RoPE，然后应用新的
                    
                    # 简化方案：不修改Q，而是调整KV的position
                    # 让KV的position与Q匹配
                    
                    # 更好的方案：使用相对position编码
                    # 暂时保持现状，观察效果
                    pass
                
                # Adjust attention mask
                attention_mask = SparseAttentionCompressor.adjust_attention_mask(
                    attention_mask=attention_mask,
                    num_anchors=num_anchors,
                    window_size=anchor_store.window_size,
                    original_seq_len=original_seq_len,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
                
                # Performance logging (降低日志频率，每50个token输出一次)
                if original_seq_len % 50 == 0 and original_seq_len > 0:
                    compression_ratio = key_states.shape[2] / original_seq_len
                    logger.info(
                        "[SparseAttn] KV compressed: %s -> %s (ratio: %.1f%%)",
                        original_seq_len,
                        key_states.shape[2],
                        compression_ratio * 100,
                    )

            # Continue with second half of original forward
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
        
        # Apply patch
        AttentionClass.forward = patched_forward
        
        store = get_memory_anchor_store()
        print(f"[QwenNarrowBandPatch] [OK] Successfully patched {AttentionClass.__name__} (true sparse attention)")
        print(f"  - Memory anchor support: enabled")
        print(f"  - Max anchors: {store.max_anchors}")
        print(f"  - Window size: {store.window_size}")
        print(f"  - Complexity: O(n * (window+anchors)) instead of O(n^2)")
        
        return True
        
    except Exception as e:
        print(f"[QwenNarrowBandPatch] [FAIL] Patch failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Helper functions from transformers
def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings (supports Qwen3.5 and Qwen2)"""
    for module_path in [
        'transformers.models.qwen3_5.modeling_qwen3_5',
        'transformers.models.qwen2.modeling_qwen2',
    ]:
        try:
            mod = __import__(module_path, fromlist=['apply_rotary_pos_emb'])
            return mod.apply_rotary_pos_emb(q, k, cos, sin)
        except (ImportError, AttributeError):
            continue
    # Fallback: manual implementation
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half"""
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
    Standard attention forward pass
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


# Try to apply patch on module import
# auto_patch()  # Uncomment to auto-apply on import
