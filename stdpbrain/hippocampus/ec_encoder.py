"""
EC (Entorhinal Cortex) 内嗅皮层 - 特征编码单元

功能:
- 接收模型注意力层输出的 token 特征
- 归一化稀疏编码为 64 维固定低维特征向量
- 每个刷新周期同步执行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EntorhinalEncoder(nn.Module):
    """
    内嗅皮层特征编码器
    
    将高维 token 特征压缩为低维特征向量
    模拟生物脑内嗅皮层的感觉信息压缩功能
    """
    def __init__(
        self,
        input_dim: int = 896,   # 模型 hidden_size (由调用方传入, 默认 Qwen2.5-0.5B)
        output_dim: int = 256,  # EC 编码维度 (由 HippocampusConfig.EC_feature_dim 控制)
        sparsity: float = 0.8,  # 稀疏度
        freeze_encoder: bool = False  # 默认允许训练，使编码器能适应模型特征空间
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        
        # ========== 编码网络 ==========
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        # ========== 稀疏掩码 ==========
        self.register_buffer(
            'sparse_mask',
            torch.ones(output_dim)
        )
        self._generate_sparse_mask()
        
        # ========== 是否冻结 ==========
        if freeze_encoder:
            self._freeze_parameters()
    
    def _freeze_parameters(self):
        """冻结编码器参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def _generate_sparse_mask(self):
        """生成稀疏掩码 (随机抑制部分神经元)"""
        num_active = int(self.output_dim * (1 - self.sparsity))
        self.sparse_mask.zero_()
        indices = torch.randperm(self.output_dim)[:num_active]
        self.sparse_mask[indices] = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征编码
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim] 或 [batch_size, input_dim]
        
        Returns:
            ec_code: EC 编码特征 [batch_size, seq_len, output_dim] 或 [batch_size, output_dim]
        """
        original_shape = x.shape
        
        # 处理形状
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, dim]
        
        batch_size, seq_len, _ = x.shape
        
        # ========== 1. 编码 (自动匹配输入dtype) ==========
        # EC编码器参数通常是FP32, 模型输出可能是FP16
        ec_code = self.encoder(x.to(self.encoder[0].weight.dtype))
        
        # ========== 2. 归一化 ==========
        ec_code = F.normalize(ec_code, p=2, dim=-1)
        
        # ========== 3. 稀疏化 ==========
        # 自动匹配输入的数据类型 (FP16/FP32)
        sparse_mask = self.sparse_mask.to(dtype=ec_code.dtype, device=ec_code.device)
        ec_code = ec_code * sparse_mask.unsqueeze(0).unsqueeze(0)
        
        # 恢复原始形状
        if len(original_shape) == 2:
            ec_code = ec_code.squeeze(1)
        
        return ec_code
    
    def encode_single(self, token_features: torch.Tensor) -> torch.Tensor:
        """
        编码单个 token 特征
        
        Args:
            token_features: [input_dim]
        
        Returns:
            ec_code: [output_dim]
        """
        if token_features.dim() == 1:
            token_features = token_features.unsqueeze(0)
        
        ec_code = self.forward(token_features)
        return ec_code.squeeze(0)
