"""
真正自指循环模块

实现冯·诺依曼式的自指：系统可以将自己的当前状态作为输入，
生成「关于这个状态」的元表示，并递归嵌套（有限深度）。

关键创新：不再有「状态」和「关于状态的表示」的二元对立，
而是让状态本身就包含对自身的指涉。
"""

import torch
import torch.nn as nn
from typing import Optional
import math

class TrueSelfReferentialLoop(nn.Module):
    """
    真正的自指循环
    
    设计原理：
    1. 基础层：原始隐状态 h_t（来自模型）
    2. 自指层：Self(h_t) → 生成关于 h_t 的表示
    3. 元层：Meta(h_t ⊕ Self(h_t)) → 整合后的自指状态
    4. 递归：可对 Meta 再次应用 Self 算子（深度=3）
    
    数学形式：
    h_0 = base_state
    h_{n+1} = Meta( h_n ⊕ Self(h_n) )
    
    其中 ⊕ 是拼接，Self 和 Meta 是可学习线性变换。
    """
    
    def __init__(self, hidden_size: int, max_recursion_depth: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_recursion_depth
        
        # 自指算子：将状态映射到「关于该状态的抽象表示」
        # 权重矩阵接近单位矩阵的扰动，保证信息不丢失
        self.self_reference = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            # 初始化为小扰动，接近恒等映射
            self.self_reference.weight.data = torch.eye(hidden_size) * 0.95 + torch.randn(hidden_size, hidden_size) * 0.01
        
        # 元层：融合原始状态和自指特征
        self.meta_layer = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        nn.init.xavier_uniform_(self.meta_layer.weight, gain=1.0)
        nn.init.constant_(self.meta_layer.bias, 0)
        
        # 递归深度衰减因子（防止无限递归的梯度爆炸）
        self.recursion_gamma = nn.Parameter(torch.tensor(0.8), requires_grad=False)
        
        # 自指强度门控（动态调整自指程度）
        self.self_gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        base_state: torch.Tensor,
        current_mind_state: str = "FOCUSED",  # 来自 InnerThoughtEngine
        recursion_depth: int = 0
    ) -> torch.Tensor:
        """
        前向传播：递归构建自指表示
        
        Args:
            base_state: 原始隐状态 [batch, hidden]
            current_mind_state: 当前思维状态（影响自指强度）
            recursion_depth: 当前递归深度（内部使用）
            
        Returns:
            self_referential_state: 嵌入了自指的最终状态
        """
        if recursion_depth >= self.max_depth:
            return base_state
        
        batch_size = base_state.shape[0]
        
        # 1. 计算自指特征
        self_ref = self.self_reference(base_state)
        
        # 2. 根据思维状态调整自指强度
        # 在 REFLECTING（反思）状态下，自指强度最大
        mind_state_weights = {
            "FOCUSED": 0.3,
            "WANDERING": 0.1,
            "REFLECTING": 0.9,  # 反思时强烈自指
            "RESTING": 0.2
        }
        state_weight = mind_state_weights.get(current_mind_state, 0.5)
        
        gate_signal = self.self_gate(base_state.mean(dim=-1, keepdim=True))  # [batch, 1]
        gate_signal = gate_signal * state_weight
        
        # 3. 拼接并应用元层
        meta_input = torch.cat([base_state, self_ref], dim=-1)
        meta_state = self.meta_layer(meta_input)
        
        # 4. 应用门控和衰减
        meta_state = meta_state * gate_signal + base_state * (1 - gate_signal)
        
        # 5. 递归（对元状态再次自指）
        if recursion_depth < self.max_depth - 1:
            # 递归时衰减，避免无限循环
            meta_state = meta_state * self.recursion_gamma + base_state * (1 - self.recursion_gamma)
            return self.forward(meta_state, current_mind_state, recursion_depth + 1)
        
        return meta_state
    
    def get_self_reference_strength(self, base_state: torch.Tensor) -> float:
        """诊断：当前自指强度（用于调试）"""
        with torch.no_grad():
            gate_val = self.self_gate(base_state.mean(dim=-1, keepdim=True))
            return gate_val.item()
