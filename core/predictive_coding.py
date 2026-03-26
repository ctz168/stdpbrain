"""
预测编码模块

实现两层预测：
1. 隐状态预测：基于当前隐藏状态预测下一时刻状态（时序模型）
2. 观测预测：基于预测的隐状态预测下一输出 token（生成模型）

核心指标：预测误差 = 实际 vs 预测的差异
使用场景：
- 误差大 → 主动提问（「我没理解，你是说...？」）
- 误差中等 → 增强 STDP 的 LTD（削弱错误关联）
- 误差小 → 增强 STDP 的 LTP（巩固正确路径）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

class PredictiveCodingModule(nn.Module):
    """预测编码主模块"""
    
    def __init__(
        self, 
        hidden_size: int,
        vocab_size: int = 50257,  # Qwen tokenizer 词汇量
        pred_hidden_size: int = 512,
        num_layers: int = 1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # 状态转移模型：GRU 预测 h_{t+1}
        # 输入：[当前隐状态, 上一时刻输出的 token embedding]
        self.state_transition = nn.GRU(
            input_size=hidden_size + hidden_size,  # h_t + token_emb
            hidden_size=pred_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # 观测模型：从预测的隐状态预测下一 token
        self.observation = nn.Sequential(
            nn.Linear(pred_hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # 预测误差历史（用于自适应阈值）
        self.error_history = []
        self.max_history_len = 100
        
        # 误差统计
        self.total_steps = 0
        self.cumulative_error = 0.0
        
    def predict_next(
        self, 
        current_state: torch.Tensor,  # [batch, hidden]
        last_output_embedding: torch.Tensor  # [batch, hidden]（上一轮生成的 token embedding）
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测下一时刻的状态和输出 token
        
        Args:
            current_state: 当前隐状态 h_t
            last_output_embedding: 上一轮输出的 token embedding（如果首轮，用零向量）
            
        Returns:
            pred_next_state: 预测的下一隐状态 h_{t+1} [batch, pred_hidden]
            pred_token_logits: 预测的下一 token 分布 [batch, vocab]
        """
        # 拼接输入
        gru_input = torch.cat([current_state, last_output_embedding], dim=-1)
        gru_input = gru_input.unsqueeze(1)  # [batch, 1, 2*hidden]
        
        # 预测下一状态
        pred_next_state, _ = self.state_transition(gru_input)
        pred_next_state = pred_next_state.squeeze(1)  # [batch, pred_hidden]
        
        # 预测下一 token
        pred_token_logits = self.observation(pred_next_state)
        
        return pred_next_state, pred_token_logits
    
    def compute_prediction_error(
        self,
        predicted_logits: torch.Tensor,
        actual_token_ids: torch.Tensor,  # [batch] 或 [batch, 1]
        predicted_state: Optional[torch.Tensor] = None,
        actual_next_state: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        计算多维度预测误差
        
        返回：
        - token_error: token 预测的交叉熵（越小越好）
        - state_error: 状态预测的 MSE（可选，需要 actual_next_state）
        - combined_error: 加权综合误差
        """
        # 1. Token 预测误差（核心）
        token_loss = F.cross_entropy(
            predicted_logits, 
            actual_token_ids.squeeze(-1) if actual_token_ids.dim() > 1 else actual_token_ids,
            reduction='mean'
        )
        token_error = token_loss.item()
        
        # 2. 状态预测误差（如果提供了实际下一状态）
        state_error = 0.0
        if predicted_state is not None and actual_next_state is not None:
            state_error = F.mse_loss(predicted_state, actual_next_state).item()
        
        # 3. 综合误差（加权平均）
        combined_error = token_error + 0.1 * state_error
        
        # 更新统计
        self.total_steps += 1
        self.cumulative_error += combined_error
        self.error_history.append(combined_error)
        if len(self.error_history) > self.max_history_len:
            self.error_history.pop(0)
        
        return {
            "token_error": token_error,
            "state_error": state_error,
            "combined_error": combined_error,
            "avg_error_recent": sum(self.error_history) / len(self.error_history),
            "error_trend": self._compute_error_trend()  # 误差上升/下降趋势
        }
    
    def _compute_error_trend(self) -> str:
        """计算误差趋势（用于自适应阈值）"""
        if len(self.error_history) < 10:
            return "stable"
        
        recent = self.error_history[-10:]
        older = self.error_history[-20:-10] if len(self.error_history) >= 20 else recent
        
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        
        if avg_recent > avg_older * 1.2:
            return "increasing"  # 误差在上升，可能模型不适应
        elif avg_recent < avg_older * 0.8:
            return "decreasing"  # 误差在下降，学习有效
        else:
            return "stable"
    
    def should_trigger_clarification(
        self, 
        current_error: float, 
        context: Dict[str, any]
    ) -> Tuple[bool, str]:
        """
        判断是否应该触发主动澄清（主动提问）
        
        Args:
            current_error: 当前预测误差
            context: 上下文（包括用户输入长度、对话历史等）
            
        Returns:
            (should_clarify, reason): 是否澄清，及原因
        """
        # 1. 动态阈值：基于历史误差自适应
        if len(self.error_history) >= 20:
            dynamic_threshold = (
                sum(self.error_history[-20:]) / 20
            ) * 1.5  # 误差超过近期平均的 50%
        else:
            dynamic_threshold = 3.0  # 默认阈值（交叉熵）
        
        # 2. 多条件判断
        is_high_error = current_error > dynamic_threshold
        is_ambiguous_input = len(context.get("user_input", "")) < 5  # 输入过短
        is_new_topic = context.get("is_new_topic", False)  # 新话题
        recent_clarifications = context.get("recent_clarifications", 0)
        is_frequent_clarifier = recent_clarifications > 2  # 过去3轮已多次澄清
        
        # 决策树
        if is_high_error and is_ambiguous_input and not is_frequent_clarifier:
            return True, f"high_error_{current_error:.2f}_ambiguous"
        
        if is_high_error and is_new_topic:
            return True, f"high_error_new_topic"
        
        return False, "error_low_or_context_ok"
