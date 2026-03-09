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
        dynamic_init_std: float = 0.01,
        static_ratio: float = 0.9
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.static_ratio = static_ratio
        
        # ========== 90% 静态基础分支 (冻结) ==========
        self.static_weight = nn.Parameter(
            torch.zeros(out_features, in_features), 
            requires_grad=False  # 永久冻结
        )
        
        if static_weight is not None:
            # 继承官方预训练权重的 90%
            self.static_weight.data = static_weight.clone() * static_ratio
        else:
            # 随机初始化 (仅用于测试)
            self.static_weight.data = torch.randn(out_features, in_features) * static_ratio
        
        # ========== 10% STDP动态增量分支 (可更新) ==========
        self.dynamic_weight = nn.Parameter(
            torch.randn(out_features, in_features) * dynamic_init_std,
            requires_grad=True  # 可学习
        )
        
        # ========== 偏置 ==========
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：总权重 = 静态 + 动态"""
        total_weight = self.static_weight + self.dynamic_weight
        return F.linear(x, total_weight, self.bias)
    
    def get_static_weight(self) -> torch.Tensor:
        """获取静态权重 (用于保存/验证)"""
        return self.static_weight.clone()
    
    def get_dynamic_weight(self) -> torch.Tensor:
        """获取动态权重 (用于 STDP 更新)"""
        return self.dynamic_weight.clone()
    
    def apply_stdp_update(self, delta_w: torch.Tensor, lr: float = 0.01):
        """
        应用 STDP 权重更新
        
        Args:
            delta_w: 权重更新量 (与 dynamic_weight 同形状)
            lr: 学习率
        """
        with torch.no_grad():
            self.dynamic_weight.add_(delta_w * lr)
            
            # 权重裁剪，防止爆炸
            self.dynamic_weight.clamp_(-1.0, 1.0)
    
    def reset_dynamic_weight(self):
        """重置动态权重 (用于初始化或清除学习)"""
        nn.init.normal_(self.dynamic_weight, mean=0, std=0.01)


class DualWeightAttention(nn.Module):
    """
    双权重自注意力机制
    
    在 Qwen3.5-0.8B 的 Multi-Head Attention 基础上，
    将 Q/K/V/O 四个线性层全部替换为 DualWeightLinear
    """
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
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            memory_anchor: 海马体记忆锚点 [batch_size, num_anchors, hidden_size]
            output_attentions: 是否输出注意力权重
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
            attn_weights: (optional) [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # ========== 1. QKV 投影 ==========
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # ========== 2. 多头拆分 ==========
        query = self._reshape_heads(query, batch_size)
        key = self._reshape_heads(key, batch_size)
        value = self._reshape_heads(value, batch_size)
        
        # ========== 3. 窄窗口注意力 (O(1) 复杂度) ==========
        # 如果提供了海马体记忆锚点，仅关注锚点对应的位置
        if memory_anchor is not None and self.hippocampus_gate is not None:
            # 使用海马体门控聚焦到 1-2 个关键位置
            gate_mask = self.hippocampus_gate(query, key, memory_anchor)
            if attention_mask is not None:
                attention_mask = attention_mask + gate_mask
            else:
                attention_mask = gate_mask
        
        # ========== 4. 注意力计算 ==========
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # ========== 5. 加权求和 ==========
        attn_output = torch.matmul(attn_weights, value)
        
        # ========== 6. 多头合并 ==========
        attn_output = self._merge_heads(attn_output, batch_size, seq_len)
        
        # ========== 7. 输出投影 ==========
        output = self.o_proj(attn_output)
        
        if output_attentions:
            return output, attn_weights
        else:
            return output, None
    
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
        """对所有 QKV/O 层应用 STDP 更新"""
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
