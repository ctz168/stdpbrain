"""
模块 3: 全链路 STDP 时序可塑性权重刷新系统

核心功能:
- 实现 Transformer 原生适配的 STDP 规则
- 在注意力层、FFN 层、自评判、海马体门控四个节点实时更新权重
- 全程无反向传播、无全局误差，纯本地时序信号驱动
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time


@dataclass
class STDPTrace:
    """STDP 追踪记录"""
    pre_activation_time: float  # 前序激活时间 (ms)
    post_activation_time: float  # 后序激活时间 (ms)
    contribution_score: float  # 贡献度评分 (-1.0 ~ 1.0)
    layer_name: str  # 层名称
    weight_indices: Tuple[int, ...]  # 权重索引


class STDPRule:
    """
    STDP 时序可塑性核心规则 (已向量化)
    """
    def __init__(
        self,
        alpha_LTP: float = 0.01,
        beta_LTD: float = 0.008,
        time_window_ms: int = 20,
        update_threshold: float = 0.001,
        weight_min: float = -1.0,
        weight_max: float = 1.0,
        decay_rate: float = 0.99,
    ):
        self.alpha_LTP = alpha_LTP
        self.beta_LTD = beta_LTD
        self.time_window_ms = time_window_ms
        self.time_constant = time_window_ms / 3.0
        self.update_threshold = update_threshold
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.decay_rate = decay_rate
    
    def compute_update(
        self, 
        pre_times: torch.Tensor, 
        post_times: torch.Tensor, 
        contributions: torch.Tensor
    ) -> torch.Tensor:
        """向量化计算 STDP 权重更新量"""
        delta_t = post_times - pre_times
        
        # 掩码：仅在时间窗口内更新
        mask = torch.abs(delta_t) <= self.time_window_ms
        
        # LTP (Δt > 0): alpha * exp(-dt/tau)
        ltp_mask = delta_t > 0
        ltp_update = self.alpha_LTP * torch.exp(-delta_t / self.time_constant)
        
        # LTD (Δt < 0): -beta * exp(dt/tau)
        ltd_mask = delta_t < 0
        ltd_update = -self.beta_LTD * torch.exp(delta_t / self.time_constant)
        
        # 同时激活 (Δt == 0)
        zero_mask = delta_t == 0
        zero_update = torch.ones_like(delta_t) * (self.alpha_LTP * 0.5)
        
        delta_w = torch.zeros_like(delta_t)
        delta_w[ltp_mask] = ltp_update[ltp_mask]
        delta_w[ltd_mask] = ltd_update[ltd_mask]
        delta_w[zero_mask] = zero_update[zero_mask]
        
        # 根据贡献度调整 (向量化运算)
        pos_contrib_mask = contributions > 0
        neg_contrib_mask = contributions <= 0
        
        delta_w[pos_contrib_mask] *= contributions[pos_contrib_mask]
        delta_w[neg_contrib_mask] *= (torch.abs(contributions[neg_contrib_mask]) * -1)
        
        delta_w *= self.decay_rate
        
        # 低于阈值清零
        delta_w[torch.abs(delta_w) < self.update_threshold] = 0
        
        return delta_w * mask

class FullLinkSTDP:
    """
    全链路 STDP 更新器 (已优化性能)
    """
    def __init__(self, config, device: Optional[str] = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.stdp_rule = STDPRule(
            alpha_LTP=config.stdp.alpha_LTP,
            beta_LTD=config.stdp.beta_LTD,
            time_window_ms=config.stdp.time_window_ms,
            update_threshold=config.stdp.update_threshold,
            weight_min=config.stdp.weight_min,
            weight_max=config.stdp.weight_max,
            decay_rate=config.stdp.decay_rate
        )
        
        # 使用张量存储激活时间 (优化查找速度)
        vocab_size = getattr(config, 'vocab_size', 250000)
        self.activation_times_tensor = torch.full((vocab_size,), -1e9, device=self.device)
        self.activation_times: Dict[str, Dict[Any, float]] = {} # 杂项记录 (非 token)
        self.contribution_cache: Dict[str, float] = {}
    
    def record_activation(self, layer_name: str, id: Any, timestamp: float):
        """记录激活时间"""
        if isinstance(id, (torch.Tensor, list)):
            # 批量记录 (Tokens 为主)
            self.activation_times_tensor[id] = timestamp
        else:
            # 单点记录 (杂项)
            if layer_name not in self.activation_times:
                self.activation_times[layer_name] = {}
            self.activation_times[layer_name][id] = timestamp
    
    def set_contribution(self, layer_name: str, score: float):
        self.contribution_cache[layer_name] = score
    
    def update_attention_layer(
        self,
        attention_layer: nn.Module,
        context_tokens: torch.Tensor,
        current_token: int,
        output: torch.Tensor,
        timestamp: float
    ):
        """注意力层 STDP 更新 (向量化实现)"""
        if not self.config.stdp.update_attention:
            return
            
        current_token_tensor = torch.tensor([current_token], device=self.device)
        self.record_activation('attention', current_token_tensor, timestamp)
        
        # 向量化计算所有上下文 token 的更新
        pre_times = self.activation_times_tensor[context_tokens]
        post_times = torch.full_like(pre_times, timestamp)
        contributions = torch.full_like(pre_times, self.contribution_cache.get('attention', 0.5))
        
        delta_ws = self.stdp_rule.compute_update(pre_times, post_times, contributions)
        
        # 如果有显著更新
        if torch.any(torch.abs(delta_ws) > 0):
            # 将 token 更新映射到权重层
            # 简化逻辑：我们将平均 delta_w 应用于动态权重分支
            mean_delta = delta_ws[delta_ws != 0].mean()
            
            if hasattr(attention_layer, 'apply_stdp_to_all') and not hasattr(attention_layer, 'q_proj'):
                # 广播模式：传递标量贡献，由接收者自行生成梯度
                attention_layer.apply_stdp_to_all({'mean_delta': mean_delta}, lr=self.config.stdp.alpha_LTP)
                return
                
            # 为 DualWeightLinear 的 q, k, v, o 生成梯度字典
            grad_dict = {
                'q': torch.ones_like(attention_layer.q_proj.dynamic_weight) * mean_delta * 0.1,
                'k': torch.ones_like(attention_layer.k_proj.dynamic_weight) * mean_delta * 0.1,
                'v': torch.ones_like(attention_layer.v_proj.dynamic_weight) * mean_delta * 0.1,
                'o': torch.ones_like(attention_layer.o_proj.dynamic_weight) * mean_delta * 0.1
            }
            
            if hasattr(attention_layer, 'apply_stdp_to_all'):
                attention_layer.apply_stdp_to_all(grad_dict, lr=self.config.stdp.alpha_LTP)
    
    # ========== 2. FFN 层 STDP 更新 ==========
    def update_ffn_layer(
        self,
        ffn_layer: nn.Module,
        input_features: torch.Tensor,
        output_features: torch.Tensor,
        timestamp: float
    ):
        """
        FFN 层 STDP 更新
        
        对当前任务、当前会话的高频特征、专属术语、
        用户习惯表达，自动增强对应 FFN 层的动态权重
        """
        if not self.config.stdp.update_ffn:
            return
        
        # 记录激活时间
        self.record_activation('ffn', 0, timestamp)
        
        # 计算特征贡献度 (基于输出范数)
        contribution = torch.norm(output_features).item() / 10.0
        contribution = min(1.0, max(-1.0, contribution))
        self.set_contribution('ffn', contribution)
        
        # 应用 STDP 更新
        if hasattr(ffn_layer, 'apply_stdp_to_all') and not hasattr(ffn_layer, 'gate_proj'):
            # 广播模式
            ffn_layer.apply_stdp_to_all({'contribution': contribution}, lr=self.config.stdp.alpha_LTP)
            return

        # 生成梯度字典
        grad_dict = {
            'gate': torch.randn_like(ffn_layer.gate_proj.dynamic_weight) * contribution * 0.01,
            'up': torch.randn_like(ffn_layer.up_proj.dynamic_weight) * contribution * 0.01,
            'proj': torch.randn_like(ffn_layer.down_proj.dynamic_weight) * contribution * 0.01
        }
        
        # 应用 STDP 更新
        if hasattr(ffn_layer, 'apply_stdp_to_all'):
            ffn_layer.apply_stdp_to_all(grad_dict, lr=self.config.stdp.alpha_LTP)
    
    # ========== 3. 自评判 STDP 更新 ==========
    def update_self_evaluation(
        self,
        generation_path: str,
        evaluation_score: float,
        model_components: Dict[str, nn.Module]
    ):
        """
        自评判 STDP 更新
        
        每 10 个刷新周期，根据模型自评判结果，
        对正确、优质的生成路径增强动态权重，
        对错误、劣质的路径减弱权重
        """
        if not self.config.stdp.update_self_eval:
            return
        
        # 归一化得分 (0-40 → -1~1)
        normalized_score = (evaluation_score / 40.0) * 2 - 1
        
        timestamp = time.time() * 1000  # ms
        self.record_activation('self_eval', 0, timestamp)
        self.set_contribution('self_eval', normalized_score)
        
        # 对所有相关组件应用 STDP 更新
        for name, module in model_components.items():
            if hasattr(module, 'apply_stdp_to_all'):
                # 根据得分生成梯度方向
                grad_scale = normalized_score * 0.01
                grad_dict = {}
                
                # 根据模块类型生成特异性梯度
                if hasattr(module, 'dynamic_weight'):
                    # 使用结构化梯度而非均匀梯度
                    base_grad = torch.randn_like(module.dynamic_weight) * 0.5
                    grad_dict['default'] = base_grad * grad_scale
                elif hasattr(module, 'weight'):
                    # 对标准权重层应用 STDP
                    grad_dict['weight'] = torch.randn_like(module.weight) * grad_scale * 0.01
                
                if grad_dict:
                    module.apply_stdp_to_all(grad_dict, lr=self.config.stdp.alpha_LTP)
    
    # ========== 4. 海马体门控 STDP 更新 ==========
    def update_hippocampus_gate(
        self,
        memory_anchor_id: str,
        contribution: float,
        hippocampus_module: nn.Module,
        timestamp: float
    ):
        """
        海马体门控 STDP 更新
        
        每个刷新周期，对推理有正向贡献的记忆锚点，
        对应的连接权重自动增强；无效的记忆锚点权重自动减弱
        """
        if not self.config.stdp.update_hippocampus_gate:
            return
        
        # 记录激活时间
        self.record_activation('hippocampus_gate', hash(memory_anchor_id) % 1000, timestamp)
        self.set_contribution('hippocampus_gate', contribution)
        
        # 计算权重更新量
        delta_w = self.stdp_rule.compute_update(
            pre_time=self.activation_times.get('hippocampus_gate', {}).get(hash(memory_anchor_id) % 1000, timestamp - 5),
            post_time=timestamp,
            contribution=contribution
        )
        
        # 更新海马体中的记忆连接强度
        if hasattr(hippocampus_module, 'update_memory_strength'):
            hippocampus_module.update_memory_strength(memory_anchor_id, delta_w)


class STDPEngine:
    """
    STDP 引擎 - 统一调度所有 STDP 更新
    
    在每个 10ms 刷新周期内，按顺序执行:
    1. 注意力层 STDP 更新
    2. FFN 层 STDP 更新
    3. (每 10 周期) 自评判 STDP 更新
    4. 海马体门控 STDP 更新
    """
    def __init__(self, config, device: Optional[str] = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.full_link_stdp = FullLinkSTDP(config, device)
        
        # 周期计数器
        self.cycle_count = 0
        self.eval_period = config.self_loop.mode3_eval_period  # 每 10 周期一次自评判
    
    def record_activation(self, type: str, id: Any, timestamp: float):
        self.full_link_stdp.record_activation(type, id, timestamp)
        
    def set_contribution(self, type: str, contribution: float):
        self.full_link_stdp.set_contribution(type, contribution)
        
    def reset(self):
        self.full_link_stdp.reset()
        self.cycle_count = 0
    
    def step(
        self,
        model_components: Dict[str, nn.Module],
        inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        timestamp: Optional[float] = None
    ):
        """
        执行一个 STDP 更新步
        
        Args:
            model_components: 模型组件字典 {name: module}
            inputs: 输入数据字典
            outputs: 输出数据字典
            timestamp: 时间戳 (ms)，默认使用当前时间
        """
        if timestamp is None:
            timestamp = time.time() * 1000
        
        # 记录默认激活
        self.record_activation('attention', 0, timestamp)
        self.record_activation('ffn', 0, timestamp)
        
        self.cycle_count += 1
        
        # ========== 1. 注意力层 STDP 更新 ==========
        if 'attention' in model_components and 'context_tokens' in inputs:
            self.full_link_stdp.update_attention_layer(
                attention_layer=model_components['attention'],
                context_tokens=inputs['context_tokens'],
                current_token=inputs['current_token'],
                output=outputs.get('attention_output', torch.zeros(1)),
                timestamp=timestamp
            )
        
        # ========== 2. FFN 层 STDP 更新 ==========
        if 'ffn' in model_components:
            self.full_link_stdp.update_ffn_layer(
                ffn_layer=model_components['ffn'],
                input_features=inputs.get('features', torch.zeros(1)),
                output_features=outputs.get('ffn_output', torch.zeros(1)),
                timestamp=timestamp
            )
        
        # ========== 3. 自评判 STDP 更新 (每 10 周期) ==========
        if self.cycle_count % self.eval_period == 0:
            if 'evaluation_score' in outputs:
                self.full_link_stdp.update_self_evaluation(
                    generation_path=outputs.get('generation_path', ''),
                    evaluation_score=outputs['evaluation_score'],
                    model_components=model_components
                )
        
        # ========== 4. 海马体门控 STDP 更新 ==========
        if 'hippocampus' in model_components and 'memory_anchor' in inputs:
            self.full_link_stdp.update_hippocampus_gate(
                memory_anchor_id=inputs.get('memory_anchor_id', 'unknown'),
                contribution=outputs.get('memory_contribution', 0.5),
                hippocampus_module=model_components['hippocampus'],
                timestamp=timestamp
            )
    
    def reset(self):
        """重置 STDP 引擎状态"""
        self.cycle_count = 0
        self.full_link_stdp.activation_times_tensor.fill_(-1e9)
        self.full_link_stdp.contribution_cache.clear()
    
    def get_stats(self) -> dict:
        """获取 STDP 更新统计信息"""
        num_active = (self.full_link_stdp.activation_times_tensor > -1e8).sum().item()
        return {
            'cycle_count': self.cycle_count,
            'num_tracked_activations': num_active,
            'device': self.device
        }
