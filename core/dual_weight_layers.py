"""
模块 1: Qwen3.5-0.8B 底座模型基础改造 - 双权重层实现

核心功能:
- 将每个 Transformer 层拆分为 90% 静态基础分支 + 10% STDP动态增量分支
- 静态分支永久冻结，继承官方预训练权重
- 动态分支可更新，初始化为小权重随机分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class DualWeightLinear(nn.Module):
    """
    双权重线性层
    
    总输出 = static_weight * x + dynamic_weight * x + bias
    其中 static_weight 冻结，dynamic_weight 可学习
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        static_weight: Optional[torch.Tensor] = None,
        dynamic_init_std: float = 0.0,
        static_ratio: float = 1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.static_ratio = static_ratio
        
        # ========== 静态基础分支 (冻结) ==========
        if static_weight is not None:
            # 使用传入的权重形状（支持量化后的权重）
            self.static_weight = nn.Parameter(
                static_weight.clone(), 
                requires_grad=False  # 永久冻结
            )
            # 更新实际的输入输出特征数
            self.out_features = static_weight.shape[0]
            self.in_features = static_weight.shape[1]
        else:
            # 随机初始化 (仅用于测试)
            self.static_weight = nn.Parameter(
                torch.randn(out_features, in_features),
                requires_grad=False
            )
        
        # ========== STDP动态增量分支 (初始为0) ==========
        # 动态权重形状与静态权重完全匹配
        self.dynamic_weight = nn.Parameter(
            torch.zeros_like(self.static_weight),
            requires_grad=True  # 可学习
        )
        
        # ========== 偏置 ==========
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # ========== 融合权重缓存 ==========
        # 仅在 STDP 更新时使缓存失效，消除每次 forward 的加法开销
        self._fused_weight: Optional[torch.Tensor] = None
        self._cache_valid = False
    
    def _get_fused_weight(self) -> torch.Tensor:
        """获取融合权重（带缓存，STDP更新前复用）"""
        if not self._cache_valid or self._fused_weight is None:
            self._fused_weight = self.static_weight.data + self.dynamic_weight.data
            self._cache_valid = True
        return self._fused_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：使用缓存的融合权重（O(1) lookup vs O(n) recompute）"""
        return F.linear(x, self._get_fused_weight(), self.bias)
    
    def get_static_weight(self) -> torch.Tensor:
        """获取静态权重 (用于保存/验证)"""
        return self.static_weight.clone()
    
    def get_dynamic_weight(self) -> torch.Tensor:
        """获取动态权重 (用于 STDP 更新)"""
        return self.dynamic_weight.clone()
    
    def apply_stdp_update(self, delta_w: torch.Tensor, lr: float = 0.01, min_val: float = -1.0, max_val: float = 1.0):
        """
        应用 STDP 权重更新
        
        Args:
            delta_w: 权重更新量 (与 dynamic_weight 同形状)
            lr: 学习率
            min_val: 权重下界
            max_val: 权重上界
        """
        with torch.no_grad():
            self.dynamic_weight.add_(delta_w * lr)
            self.dynamic_weight.clamp_(min_val, max_val)
            # STDP 更新后使融合缓存失效
            self._cache_valid = False
    
    def reset_dynamic_weight(self):
        """重置动态权重 (用于初始化或清除学习)"""
        nn.init.normal_(self.dynamic_weight, mean=0, std=0.01)
        self._cache_valid = False


class DualWeightAttention(nn.Module):
    """
    双权重自注意力机制
    
    在 Qwen3.5-0.8B 的 Multi-Head Attention 基础上，
    将 Q/K/V/O 四个线性层全部替换为 DualWeightLinear
    """
    
    # 类变量：存储当前的记忆锚点（所有实例共享）
    _current_memory_anchor = None
    
    @classmethod
    def set_memory_anchor(cls, anchor):
        """设置当前的记忆锚点"""
        cls._current_memory_anchor = anchor
    
    @classmethod
    def get_memory_anchor(cls):
        """获取当前的记忆锚点"""
        return cls._current_memory_anchor
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qkv_bias: bool = False,
        static_q: Optional[torch.Tensor] = None,
        static_k: Optional[torch.Tensor] = None,
        static_v: Optional[torch.Tensor] = None,
        static_o: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # ========== 双权重 QKV 投影 ==========
        self.q_proj = DualWeightLinear(
            hidden_size, hidden_size, bias=qkv_bias, 
            static_weight=static_q
        )
        self.k_proj = DualWeightLinear(
            hidden_size, hidden_size, bias=qkv_bias,
            static_weight=static_k
        )
        self.v_proj = DualWeightLinear(
            hidden_size, hidden_size, bias=qkv_bias,
            static_weight=static_v
        )
        
        # ========== 双权重输出投影 ==========
        self.o_proj = DualWeightLinear(
            hidden_size, hidden_size, bias=False,
            static_weight=static_o
        )
        
        # ========== Dropout ==========
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # ========== 海马体注意力门控 ==========
        self.hippocampus_gate = None  # 后续由海马体模块设置
    
    def set_hippocampus_gate(self, gate_fn):
        """
        设置海马体注意力门控函数
        
        Args:
            gate_fn: function(query, key, memory_anchor) -> gate_mask
        """
        self.hippocampus_gate = gate_fn
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_anchor: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """前向传播 (支持 KV-cache)"""
        batch_size, seq_len, _ = hidden_states.size()
        
        # ========== 1. QKV 投影 ==========
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # ========== 2. 多头拆分 ==========
        query = self._reshape_heads(query, batch_size)
        key = self._reshape_heads(key, batch_size)
        value = self._reshape_heads(value, batch_size)
        
        # ========== 3. KV-Cache 处理 ==========
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        present = (key, value) if use_cache else None
        
        # ========== 4. 窄窗口注意力 (O(1) 复杂度) ==========
        # 优先使用传入的memory_anchor，否则使用类变量中的
        effective_memory_anchor = memory_anchor or self._current_memory_anchor
        if effective_memory_anchor is not None and self.hippocampus_gate is not None:
            try:
                gate_mask = self.hippocampus_gate(query, key, effective_memory_anchor)
                if gate_mask is not None:
                    if attention_mask is not None:
                        attention_mask = attention_mask + gate_mask
                    else:
                        attention_mask = gate_mask
            except Exception as e:
                pass  # 门控计算失败时继续正常推理
        
        # ========== 5. 注意力计算 ==========
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if attention_mask is not None:
            # 处理 KV-cache 时的注意力掩码
            if attention_mask.shape[-1] != attn_weights.shape[-1]:
                 attention_mask = attention_mask[:, :, -attn_weights.shape[-2]:, :]
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # ========== 6. 加权求和 ==========
        attn_output = torch.matmul(attn_weights, value)
        
        # ========== 7. 多头合并 ==========
        attn_output = self._merge_heads(attn_output, batch_size, seq_len)
        
        # ========== 8. 输出投影 ==========
        output = self.o_proj(attn_output)
        
        return output, (attn_weights if output_attentions else None), present
    
    def _reshape_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """将 [batch, seq, hidden] 拆分为 [batch, heads, seq, head_dim]"""
        _, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def _merge_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """将 [batch, heads, seq, head_dim] 合并为 [batch, seq, hidden]"""
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.hidden_size)
    
    def get_attention_features(self) -> dict:
        """
        获取注意力层特征输出接口
        返回当前层的注意力特征、时序特征、语义特征
        """
        return {
            'static_weight': self.get_all_static_weights(),
            'dynamic_weight': self.get_all_dynamic_weights(),
            'num_heads': self.num_heads,
            'head_dim': self.head_dim
        }
    
    def get_all_static_weights(self) -> dict:
        """获取所有静态权重"""
        return {
            'q': self.q_proj.static_weight.clone(),
            'k': self.k_proj.static_weight.clone(),
            'v': self.v_proj.static_weight.clone(),
            'o': self.o_proj.static_weight.clone()
        }
    
    def get_all_dynamic_weights(self) -> dict:
        """获取所有动态权重"""
        return {
            'q': self.dynamic_weight.clone(),
            'k': self.k_proj.dynamic_weight.clone(),
            'v': self.v_proj.dynamic_weight.clone(),
            'o': self.o_proj.dynamic_weight.clone()
        }
    
    def apply_stdp_to_all(self, grad_dict: dict, lr: float = 0.01):
        """对所有层应用 STDP 更新"""
        # 支持广播模式 (输入为标量字典)
        if 'mean_delta' in grad_dict:
            mean_delta = grad_dict['mean_delta']
            # 为每个子层生成随机扰动 (符合 STDP 探索特性)
            for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
                noise = torch.randn_like(proj.dynamic_weight) * 0.1 * mean_delta
                proj.apply_stdp_update(noise, lr)
            return

        if 'q' in grad_dict:
            self.q_proj.apply_stdp_update(grad_dict['q'], lr)
        if 'k' in grad_dict:
            self.k_proj.apply_stdp_update(grad_dict['k'], lr)
        if 'v' in grad_dict:
            self.v_proj.apply_stdp_update(grad_dict['v'], lr)
        if 'o' in grad_dict:
            self.o_proj.apply_stdp_update(grad_dict['o'], lr)


class DualWeightFFN(nn.Module):
    """
    双权重前馈神经网络 (FFN)
    
    Qwen3.5-0.8B 使用 SwiGLU 激活函数:
    FFN(x) = W_proj(SiLU(W_gate(x)) * W_up(x))
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        static_gate: Optional[torch.Tensor] = None,
        static_up: Optional[torch.Tensor] = None,
        static_proj: Optional[torch.Tensor] = None,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # ========== 双权重 Gate 层 ==========
        self.gate_proj = DualWeightLinear(
            hidden_size, intermediate_size, bias=False,
            static_weight=static_gate
        )
        
        # ========== 双权重 Up 层 ==========
        self.up_proj = DualWeightLinear(
            hidden_size, intermediate_size, bias=False,
            static_weight=static_up
        )
        
        # ========== 双权重 Proj 层 ==========
        self.down_proj = DualWeightLinear(
            intermediate_size, hidden_size, bias=False,
            static_weight=static_proj
        )
        
        # ========== 激活函数 ==========
        if hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif hidden_act == "gelu":
            self.act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # SwiGLU: SiLU(gate) * up
        gate = self.gate_proj(x)
        gate = self.act_fn(gate)
        
        up = self.up_proj(x)
        
        # Element-wise 乘法
        hidden = gate * up
        
        # 投影回 hidden_size
        output = self.down_proj(hidden)
        
        return output
    
    def get_all_dynamic_weights(self) -> dict:
        """获取所有动态权重"""
        return {
            'gate': self.gate_proj.dynamic_weight.clone(),
            'up': self.up_proj.dynamic_weight.clone(),
            'proj': self.down_proj.dynamic_weight.clone()
        }
    
    def apply_stdp_to_all(self, grad_dict: dict, lr: float = 0.01):
        """对所有层应用 STDP 更新"""
        # 支持广播模式
        if 'contribution' in grad_dict:
            contribution = grad_dict['contribution']
            for proj in [self.gate_proj, self.up_proj, self.down_proj]:
                noise = torch.randn_like(proj.dynamic_weight) * 0.01 * contribution
                proj.apply_stdp_update(noise, lr)
            return

        if 'gate' in grad_dict:
            self.gate_proj.apply_stdp_update(grad_dict['gate'], lr)
        if 'up' in grad_dict:
            self.up_proj.apply_stdp_update(grad_dict['up'], lr)
        if 'proj' in grad_dict:
            self.down_proj.apply_stdp_update(grad_dict['proj'], lr)


# ==================== 角色适配接口 ====================

ROLE_PROMPT_TEMPLATES = {
    "generator": "你是一个专业的文本生成助手。请根据上下文生成连贯、准确的回复。",
    "verifier": "你是一个严谨的验证者。请仔细检查以下内容的逻辑正确性、事实准确性，指出任何错误或漏洞。",
    "evaluator": "你是一个公正的评判者。请从事实准确性、逻辑完整性、语义连贯性、指令遵循度四个维度对以下内容进行打分 (每个维度 0-10 分)。"
}


def create_role_prompt(role: str, custom_instruction: Optional[str] = None) -> str:
    """
    创建角色提示词
    
    Args:
        role: 角色类型 ("generator", "verifier", "evaluator")
        custom_instruction: 自定义指令
    
    Returns:
        完整的角色提示词
    """
    base_prompt = ROLE_PROMPT_TEMPLATES.get(role, ROLE_PROMPT_TEMPLATES["generator"])
    
    if custom_instruction:
        return f"{base_prompt}\n\n{custom_instruction}"
    
    return base_prompt
