"""
模块 1: Qwen3.5-2B 底座模型基础改造 - 双权重层实现

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
    双权重线性层 (LoRA 分支流架构)
    
    总输出 = base_layer(x) + dynamic_weight * x
    完美兼容 4-bit / 8-bit 量化底座
    """
    def __init__(
        self, 
        base_layer: nn.Module,
        dynamic_init_std: float = 0.0,
        static_ratio: float = 1.0
    ):
        super().__init__()
        self.base_layer = base_layer
        self.static_ratio = static_ratio
        
        # 取尺寸参数
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # 从底座推断设备
        try:
            sample_param = next(base_layer.parameters())
            target_device = sample_param.device
            # 动态层必须保持高精度应对微小 STDP 梯度
            target_dtype = torch.float16 if target_device.type == "cuda" else torch.float32
        except StopIteration:
            target_device = torch.device('cpu')
            target_dtype = torch.float32
            
        # ========== STDP动态增量分支 (初始为0) ==========
        self.dynamic_weight = nn.Parameter(
            torch.zeros(self.out_features, self.in_features, 
                       device=target_device, dtype=target_dtype),
            requires_grad=True  # 可学习
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：基座推理 + STDP 增量并行计算"""
        # 1. 静态计算 (量化黑盒或其他各种形式的底层计算)
        base_out = self.base_layer(x)
        
        # 2. 动态增量计算 (确保使用 FP16/FP32 计算保持精度)
        x_dyn = x.to(self.dynamic_weight.dtype)
        dynamic_out = F.linear(x_dyn, self.dynamic_weight)
        
        # 3. 输出聚合，还原回基底的数据类型(防止外层发生类型冲突)
        out = (base_out * self.static_ratio) + dynamic_out.to(base_out.dtype)
        return out
     
    def get_dynamic_weight(self) -> torch.Tensor:
        """获取动态权重 (用于 STDP 更新)"""
        return self.dynamic_weight.clone()
    
    def apply_stdp_update(self, delta_w: torch.Tensor, lr: float = 0.01, min_val: float = -1.0, max_val: float = 1.0):
        """应用 STDP 权重更新"""
        with torch.no_grad():
            # 确保数据类型一致，防止 CUDA 半精度报错
            if delta_w.dtype != self.dynamic_weight.dtype:
                delta_w = delta_w.to(self.dynamic_weight.dtype)
            self.dynamic_weight.add_(delta_w * lr)
            self.dynamic_weight.clamp_(min_val, max_val)


