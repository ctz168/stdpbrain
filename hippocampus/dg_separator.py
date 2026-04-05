"""
DG (Dentate Gyrus) 齿状回 - 模式分离单元

功能:
- 对 EC 编码特征做稀疏随机投影正交化处理
- 为相似输入生成完全正交的唯一记忆 ID
- 从根源避免记忆混淆
- 无训练参数 (纯数学变换)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import hashlib


class DentateGyrusSeparator(nn.Module):
    """
    齿状回模式分离器
    
    通过稀疏随机投影将相似的 EC 编码转换为正交的记忆 ID
    模拟生物齿状回的模式分离功能，避免记忆混淆
    """
    def __init__(
        self,
        input_dim: int = 64,       # EC 编码维度
        output_dim: int = 128,     # DG 输出维度 (扩展以增加正交性)
        sparsity: float = 0.7,     # 稀疏度 (从0.9降到0.7，保留更多语义信息)
        orthogonalization: bool = True,  # 启用正交化
        learnable_projection: bool = False  # 是否使用可学习的投影矩阵
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.orthogonalization = orthogonalization
        self.learnable_projection = learnable_projection
        
        # ========== 稀疏随机投影矩阵 ==========
        # 使用稀疏随机矩阵实现模式分离
        projection = torch.randn(output_dim, input_dim) / np.sqrt(input_dim)
        
        # 应用稀疏掩码
        mask = (torch.rand_like(projection) > sparsity).float()
        projection = projection * mask
        
        # 防止全零行导致 NaN（F.normalize 对全零向量产生 NaN）
        zero_rows = (projection.abs().sum(dim=1) == 0)
        if zero_rows.any():
            # 用新的随机行替换全零行
            num_zero = zero_rows.sum().item()
            projection[zero_rows] = F.normalize(
                torch.randn(num_zero, input_dim) / np.sqrt(input_dim), p=2, dim=1
            )
        
        # 归一化每行
        projection = F.normalize(projection, p=2, dim=1)
        
        if learnable_projection:
            # 可学习投影矩阵：使用nn.Parameter，允许通过反向传播优化
            self.projection_matrix = nn.Parameter(projection)
        else:
            # 固定投影矩阵：使用register_buffer
            self.register_buffer('projection_matrix', projection)
        
        # ========== 正交化基 (可选) ==========
        if self.orthogonalization:
            # 生成一组近似正交的基向量
            orthogonal_basis = self._generate_orthogonal_basis()
            self.register_buffer('orthogonal_basis', orthogonal_basis)
    
    def _generate_orthogonal_basis(self) -> torch.Tensor:
        """生成近似正交的基向量 (Gram-Schmidt 过程)"""
        # 随机初始化
        basis = torch.randn(self.output_dim, self.output_dim)
        
        # Gram-Schmidt 正交化
        for i in range(self.output_dim):
            for j in range(i):
                basis[i] -= torch.dot(basis[i], basis[j]) * basis[j]
            basis[i] = F.normalize(basis[i], p=2, dim=0)
        
        return basis
    
    def forward(self, ec_code: torch.Tensor) -> torch.Tensor:
        """
        模式分离
        
        Args:
            ec_code: EC 编码特征 [batch_size, seq_len, input_dim] 或 [input_dim]
        
        Returns:
            dg_output: DG 分离后的特征 [batch_size, seq_len, output_dim] 或 [output_dim]
        """
        original_shape = ec_code.shape
        
        # 处理形状
        if ec_code.dim() == 1:
            ec_code = ec_code.unsqueeze(0)  # [1, output_dim]
        
        if ec_code.dim() == 2:
            ec_code = ec_code.unsqueeze(1)  # [batch, 1, output_dim]
        
        batch_size, seq_len, _ = ec_code.shape
        
        # ========== 1. 稀疏随机投影 ==========
        # 自动匹配输入的数据类型 (FP16/FP32)
        proj_matrix = self.projection_matrix.to(dtype=ec_code.dtype, device=ec_code.device)
        dg_output = F.linear(ec_code, proj_matrix)  # [batch, seq, output_dim]
        
        # ========== 2. 非线性激活 (增强分离效果) ==========
        dg_output = F.relu(dg_output)
        
        # ========== 3. 正交化 (可选) ==========
        if self.orthogonalization:
            # 将输出投影到正交基上
            orth_basis = self.orthogonal_basis.to(dtype=dg_output.dtype, device=dg_output.device)
            dg_output = torch.matmul(dg_output, orth_basis.t())
        
        # ========== 4. 归一化 ==========
        dg_output = F.normalize(dg_output, p=2, dim=-1)
        
        # 恢复原始形状
        if len(original_shape) == 1:
            dg_output = dg_output.squeeze(0).squeeze(0)
            # 防止过度坍缩（当 output_dim == 1 时 squeeze 会变成标量）
            if dg_output.dim() == 0:
                dg_output = dg_output.unsqueeze(0)
        elif len(original_shape) == 2:
            dg_output = dg_output.squeeze(1)
        
        return dg_output
    
    def separate_and_id(self, ec_code: torch.Tensor) -> tuple:
        """
        模式分离并生成唯一记忆 ID
        
        Args:
            ec_code: EC 编码特征 [input_dim]
        
        Returns:
            dg_output: DG 分离后的特征 [output_dim]
            memory_id: 唯一记忆 ID (字符串)
        """
        with torch.no_grad():
            dg_output = self.forward(ec_code)
        
        # ========== 生成唯一记忆 ID ==========
        # 将 DG 输出量化为二进制串作为 ID
        binary_code = (dg_output > 0).cpu().numpy().astype(int)
        id_hash = hashlib.sha256(binary_code.tobytes()).hexdigest()[:16]
        
        return dg_output, f"mem_{id_hash}"
