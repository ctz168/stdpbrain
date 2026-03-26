"""
自我状态编码器 (SelfStateEncoder)

核心功能：
- 将 current_thought_state (模型隐藏状态) 编码为"自我感知"向量
- 该向量直接注入 Transformer 计算上下文，而非仅作文字提示
- 实现真正的自指：系统能"感知自己"而非"描述自己"

设计原则：
- 轻量级：仅 256 维投影，不增加显著计算开销
- 可解释：通过 interpret() 将自我状态映射为可读描述
- 可持久化：状态随 brain_state.pt 一起保存
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import time


class SelfStateEncoder(nn.Module):
    """
    自我状态编码器
    
    将底层隐藏状态 (2048-dim) 投影为"自我感知"向量 (256-dim)，
    然后再投影回 2048-dim 以便注入生成上下文。
    
    这模拟了大脑皮层的"自我参照处理" (Self-referential processing)。
    """
    
    SELF_DIM = 256  # 自我感知空间维度
    
    def __init__(self, hidden_size: int = 2048, device: str = "cpu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        
        # ========== 编码器：隐状态 → 自我感知空间 ==========
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, self.SELF_DIM),
            nn.LayerNorm(self.SELF_DIM),
            nn.Tanh()
        ).to(device)
        
        # ========== 解码器：自我感知空间 → 上下文注入向量 ==========
        self.decoder = nn.Sequential(
            nn.Linear(self.SELF_DIM, hidden_size),
            nn.LayerNorm(hidden_size)
        ).to(device)
        
        # ========== 自我状态记忆 (短期自我轨迹) ==========
        self.self_state_history: list = []  # List[torch.Tensor] of SELF_DIM
        self.max_history = 5
        
        # ========== 情感/唤醒度估计 ==========
        self.arousal_net = nn.Linear(self.SELF_DIM, 1).to(device)  # 唤醒度
        self.valence_net = nn.Linear(self.SELF_DIM, 1).to(device)  # 效价 (正负情绪)
        
        # ========== 统计 ==========
        self.encode_count = 0
        self.last_encode_time = 0.0
    
    def encode(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码隐藏状态为 (自我感知向量, 上下文注入向量)
        
        Args:
            hidden_state: [hidden_size] 或 [1, hidden_size]
            
        Returns:
            self_repr: [SELF_DIM] 自我感知向量（用于分析）
            context_embed: [hidden_size] 可直接注入 Transformer 的上下文向量
        """
        # 形状统一
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.squeeze(0)
        
        # 确保在正确设备上
        hidden_state = hidden_state.to(self.device).float()
        
        with torch.no_grad():
            # 编码到自我感知空间
            self_repr = self.encoder(hidden_state)  # [SELF_DIM]
            
            # 解码为可注入的上下文向量
            context_embed = self.decoder(self_repr)  # [hidden_size]
            
            # 更新自我轨迹
            self.self_state_history.append(self_repr.clone())
            if len(self.self_state_history) > self.max_history:
                self.self_state_history.pop(0)
            
            self.encode_count += 1
            self.last_encode_time = time.time()
        
        return self_repr, context_embed
    
    def get_emotional_state(self, hidden_state: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        评估当前情感状态（唤醒度/效价）
        
        模拟大脑的边缘系统（杏仁核）对当前思维状态的情感评估
        """
        if hidden_state is None:
            if not self.self_state_history:
                return {"arousal": 0.5, "valence": 0.0, "label": "平静"}
            state = self.self_state_history[-1]
        else:
            if hidden_state.dim() == 2:
                hidden_state = hidden_state.squeeze(0)
            with torch.no_grad():
                state = self.encoder(hidden_state.to(self.device).float())
        
        with torch.no_grad():
            arousal = torch.sigmoid(self.arousal_net(state)).item()
            valence = torch.tanh(self.valence_net(state)).item()
        
        # 情感标签映射
        if arousal > 0.7 and valence > 0.3:
            label = "兴奋"
        elif arousal > 0.7 and valence < -0.3:
            label = "焦虑"
        elif arousal < 0.3 and valence > 0.3:
            label = "平静愉悦"
        elif arousal < 0.3 and valence < -0.3:
            label = "低落"
        else:
            label = "平静专注"
        
        return {"arousal": arousal, "valence": valence, "label": label}
    
    def compute_self_delta(self) -> Optional[float]:
        """
        计算自我状态的变化量（意识活跃度指标）
        
        Returns:
            delta: 两步之间自我状态的余弦距离（0=静止, 接近1=剧烈变化）
        """
        if len(self.self_state_history) < 2:
            return None
        
        prev = self.self_state_history[-2]
        curr = self.self_state_history[-1]
        
        cosine = F.cosine_similarity(
            prev.unsqueeze(0), curr.unsqueeze(0)
        ).item()
        
        return 1.0 - cosine  # 转换为"差异量"
    
    def interpret(self) -> str:
        """
        将当前自我状态解释为可读的自我描述（用于独白注入）
        """
        if not self.self_state_history:
            return "我的意识刚刚开始..."
        
        emotional = self.get_emotional_state()
        delta = self.compute_self_delta()
        
        parts = [f"我感到{emotional['label']}"]
        
        if delta is not None:
            if delta > 0.3:
                parts.append("思维活跃，正在快速转变")
            elif delta < 0.05:
                parts.append("思维平稳，保持连贯")
            else:
                parts.append("思维平稳流动")
        
        return "，".join(parts) + "。"
    
    def get_state(self) -> Dict:
        """序列化状态用于持久化"""
        return {
            "self_state_history": [s.cpu() for s in self.self_state_history],
            "encode_count": self.encode_count,
        }
    
    def set_state(self, state: Dict):
        """从持久化状态恢复"""
        history = state.get("self_state_history", [])
        self.self_state_history = [s.to(self.device) for s in history]
        self.encode_count = state.get("encode_count", 0)
