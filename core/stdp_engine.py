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
import hashlib


def _stable_hash(s: str) -> int:
    """确定性哈希（跨进程/重启稳定，替代 Python 内置 hash()）"""
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 1000


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
        contributions: torch.Tensor,
        reward: float = 1.0
    ) -> torch.Tensor:
        """向量化计算 STDP 权重更新量 (支持奖励信号)"""
        delta_t = post_times - pre_times
        
        # 掩码：仅在时间窗口内更新
        mask = torch.abs(delta_t) <= self.time_window_ms
        
        # LTP (Δt > 0): alpha * exp(-dt/tau)
        ltp_mask = delta_t > 0
        # 奖励信号主要增强 LTP (正向学习)
        ltp_update = self.alpha_LTP * torch.exp(-delta_t / self.time_constant) * max(0.1, reward)
        
        # LTD (Δt < 0): -beta * exp(dt/tau)
        ltd_mask = delta_t < 0
        # 负奖励会增强 LTD (惩罚)
        ltd_update = -self.beta_LTD * torch.exp(delta_t / self.time_constant) * (1.1 - min(1.0, reward))
        
        # 同时激活 (Δt == 0)
        zero_mask = delta_t == 0
        zero_update = torch.ones_like(delta_t) * (self.alpha_LTP * 0.5 * reward)
        
        delta_w = torch.zeros_like(delta_t)
        delta_w[ltp_mask] = ltp_update[ltp_mask]
        delta_w[ltd_mask] = ltd_update[ltd_mask]
        delta_w[zero_mask] = zero_update[zero_mask]
        
        # 根据贡献度调整
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
        
        # 梯度张量缓存 (避免每次更新都创建新张量)
        self._grad_cache: Dict[str, torch.Tensor] = {}
        
        # 预分配 current_token 张量（避免每次调用 torch.tensor）
        self._current_token_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
        # 预分配 post_times 张量
        self._post_times_buffer = torch.zeros(100, dtype=torch.float32, device=self.device)
        # BUG FIX: _contributions_buffer 必须在 __init__ 中同步初始化。
        # 原代码只在 n_ctx > 100 时才创建此缓冲区，导致首次调用
        # update_attention_layer 且 context ≤100 tokens 时触发 AttributeError。
        self._contributions_buffer = torch.zeros(100, dtype=torch.float32, device=self.device)
    
    def record_activation(self, layer_name: str, id: Any, timestamp: float):
        """记录激活时间"""
        if isinstance(id, (torch.Tensor, list)):
            # 批量记录 (Tokens 为主) - 强制转为 long 类型，避免 float 索引错误
            if isinstance(id, torch.Tensor):
                id = id.long()
                # 限制索引在合法范围内（vocabulary 大小）
                vocab_size = self.activation_times_tensor.shape[0]
                id = id.clamp(0, vocab_size - 1)
            self.activation_times_tensor[id] = timestamp
        else:
            # 单点记录 (杂项)
            if layer_name not in self.activation_times:
                self.activation_times[layer_name] = {}
            self.activation_times[layer_name][id] = timestamp
    
    def set_contribution(self, layer_name: str, score: float):
        self.contribution_cache[layer_name] = score
    
    def _get_or_create_grad(self, cache_key: str, shape: tuple, device: torch.device) -> torch.Tensor:
        """获取或创建缓存的梯度张量 (性能优化)"""
        full_key = f"{cache_key}_{shape}_{device}"
        if full_key not in self._grad_cache:
            self._grad_cache[full_key] = torch.zeros(shape, device=device, dtype=torch.float32)
        return self._grad_cache[full_key]
    
    def update_attention_layer(
        self,
        attention_layer: nn.Module,
        context_tokens: torch.Tensor,
        current_token: int,
        output: torch.Tensor,
        timestamp: float,
        reward: float = 1.0,
        is_tool_call: bool = False
    ):
        """注意力层 STDP 更新 (性能优化版)"""
        if not self.config.stdp.update_attention or attention_layer is None:
            return
        
        # 快速跳过：如果 attention_layer 没有 dynamic_weight，直接返回
        if not hasattr(attention_layer, 'dynamic_weight') and not hasattr(attention_layer, 'apply_stdp_update'):
            return
        
        # 复用预分配张量（避免 torch.tensor 分配）
        self._current_token_tensor[0] = current_token
        self.record_activation('attention', self._current_token_tensor, timestamp)
        
        # 向量化计算所有上下文 token 的更新
        target_device = self.activation_times_tensor.device
        if isinstance(context_tokens, torch.Tensor):
            context_tokens = context_tokens.long().to(target_device).clamp(0, self.activation_times_tensor.shape[0] - 1)
        else:
            context_tokens = torch.tensor(context_tokens, dtype=torch.long, device=target_device).clamp(0, self.activation_times_tensor.shape[0] - 1)
        pre_times = self.activation_times_tensor[context_tokens]
        
        # [BUG FIX] 分别使用独立缓冲区，避免张量视图别名覆盖。
        # 原代码 post_times 和 contributions 都是 _post_times_buffer 的切片视图，
        # contributions.fill_(0.5) 会覆盖 post_times 刚填入的 timestamp，
        # 导致 STDP 时间差计算始终使用 0.5 而非真实时间戳，
        # LTP/LTD 判定完全失效。
        n_ctx = pre_times.shape[0]
        if n_ctx > self._post_times_buffer.shape[0]:
            self._post_times_buffer = torch.zeros(n_ctx * 2, dtype=torch.float32, device=self.device)
            if n_ctx > self._contributions_buffer.shape[0]:
                self._contributions_buffer = torch.zeros(n_ctx * 2, dtype=torch.float32, device=self.device)
        post_times = self._post_times_buffer[:n_ctx].fill_(timestamp)
        contributions = self._contributions_buffer[:n_ctx].fill_(self.contribution_cache.get('attention', 0.5))
        
        # 工具调用强化
        effective_reward = reward * (5.0 if is_tool_call else 1.0)
        
        delta_ws = self.stdp_rule.compute_update(pre_times, post_times, contributions, reward=effective_reward)
        
        # 如果有显著更新
        if torch.any(torch.abs(delta_ws) > 0):
            mean_delta = delta_ws[delta_ws != 0].mean()
            
                # ========== 优化: 使用缓存的梯度张量 ==========
            if hasattr(attention_layer, 'apply_stdp_update') and hasattr(attention_layer, 'dynamic_weight'):
                # 单层更新模式（最常见）
                weight_shape = attention_layer.dynamic_weight.shape
                grad = self._get_or_create_grad('attention_single', weight_shape, attention_layer.dynamic_weight.device)
                # BUG FIX: 使用与 dynamic_weight 匹配的数据类型
                grad = grad.to(attention_layer.dynamic_weight.dtype)
                grad.fill_(mean_delta.item() * 0.1)
                attention_layer.apply_stdp_update(grad, lr=self.config.stdp.alpha_LTP)
                return
                
            if hasattr(attention_layer, 'apply_stdp_to_all') and not hasattr(attention_layer, 'q_proj'):
                attention_layer.apply_stdp_to_all({'mean_delta': mean_delta}, lr=self.config.stdp.alpha_LTP)
                return
                
            # 为 DualWeightLinear 的 q, k, v, o 生成梯度
            # 优化: 使用缓存
            if hasattr(attention_layer, 'apply_stdp_to_all'):
                _proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
                if all(hasattr(attention_layer, p) and hasattr(getattr(attention_layer, p), 'dynamic_weight')
                       for p in _proj_names):
                    q_weight = attention_layer.q_proj.dynamic_weight
                    grad_cache = self._get_or_create_grad('attention_qkvo', q_weight.shape, q_weight.device)
                    grad_cache = grad_cache.to(q_weight.dtype)
                    grad_cache.fill_(mean_delta.item() * 0.1)
                    grad_dict = {k: grad_cache.clone() for k in ['q', 'k', 'v', 'o']}
                    attention_layer.apply_stdp_to_all(grad_dict, lr=self.config.stdp.alpha_LTP)
                    return
            
            # ========== 新增: 处理整个模型的情况 ==========
            # 如果传入的是整个模型，遍历所有 DualWeightLinear 层进行更新
            if hasattr(attention_layer, 'named_modules'):
                update_count = 0
                for name, module in attention_layer.named_modules():
                    if hasattr(module, 'dynamic_weight') and hasattr(module, 'apply_stdp_update'):
                        weight_shape = module.dynamic_weight.shape
                        grad = self._get_or_create_grad(f'model_{name}', weight_shape, module.dynamic_weight.device)
                        grad.fill_(mean_delta.item() * 0.1)
                        module.apply_stdp_update(grad, lr=self.config.stdp.alpha_LTP)
                        update_count += 1
                if update_count > 0:
                    return
    
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
        if not self.config.stdp.update_ffn or ffn_layer is None:
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
        # BUG FIX: 原代码使用 torch.randn_like() 生成随机噪声梯度。
        # 这不是 STDP 更新，而是随机扰动，会逐渐破坏 FFN 权重中的有意义模式。
        # 改为使用 contribution（归一化后的输出范数）作为有方向的梯度缩放因子，
        # 保持 STDP 的"贡献度越高，权重增强越强"的语义。
        contribution_val = min(1.0, max(-1.0, contribution))
        # 使用 mean delta 作为基础梯度（与注意力层 STDP 一致），方向由 contribution 决定
        base_grad_scale = contribution_val * 0.01  # 贡献度导向缩放
        
        if not (hasattr(ffn_layer, 'gate_proj')
                and hasattr(ffn_layer.gate_proj, 'dynamic_weight')
                and hasattr(ffn_layer, 'up_proj')
                and hasattr(ffn_layer.up_proj, 'dynamic_weight')
                and hasattr(ffn_layer, 'down_proj')
                and hasattr(ffn_layer.down_proj, 'dynamic_weight')
                and hasattr(ffn_layer, 'apply_stdp_to_all')):
            return

        grad_dict = {
            'gate': torch.ones_like(ffn_layer.gate_proj.dynamic_weight) * base_grad_scale,
            'up':   torch.ones_like(ffn_layer.up_proj.dynamic_weight)   * base_grad_scale,
            'proj': torch.ones_like(ffn_layer.down_proj.dynamic_weight) * base_grad_scale
        }
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
                # BUG FIX: 原代码使用 torch.randn_like() 生成随机噪声梯度，
                # 这不是 STDP 更新而是随机扰动，会逐渐破坏权重中的有意义模式。
                # 改为使用 normalized_score（归一化后的评判分数）作为有方向的
                # 梯度缩放因子，与 FFN STDP 更新逻辑一致：
                # 正分 → LTP（增强当前权重模式）
                # 负分 → LTD（削弱当前权重模式）
                grad_scale = normalized_score * 0.01
                grad_dict = {}
                
                # 根据模块类型生成特异性梯度（使用均匀梯度而非随机噪声）
                if hasattr(module, 'dynamic_weight'):
                    # 使用 uniform 填充而非 randn：正分时所有权重同向增强，
                    # 负分时所有权重同向削弱，保持 STDP 的"贡献度越高权重越强"语义
                    base_grad = torch.ones_like(module.dynamic_weight) * 0.5
                    grad_dict['default'] = base_grad * grad_scale
                elif hasattr(module, 'weight'):
                    # 对标准权重层应用 STDP
                    grad_dict['weight'] = torch.ones_like(module.weight) * grad_scale * 0.01
                
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
        
        # BUG FIX: 海马体门控 STDP 更新需要使用上一周期的激活时间（而非当前周期刚记录的）。
        # 原代码在同一函数内先 record_activation() 再立即读取，导致 pre_time == timestamp，
        # delta_t 恒为 0，STDP 永远走 zero_update 分支，海马体门控权重永远不更新。
        # 正确逻辑：先获取上一次的激活时间，然后更新为当前时间。
        pre_time_val = self.activation_times.get('hippocampus_gate', {}).get(
            _stable_hash(memory_anchor_id), timestamp - 5
        )
        # 先获取旧值，再记录新值
        self.record_activation('hippocampus_gate', _stable_hash(memory_anchor_id), timestamp)
        delta_w = self.stdp_rule.compute_update(
            pre_times=torch.tensor([pre_time_val], device=self.device, dtype=torch.float32),
            post_times=torch.tensor([timestamp], device=self.device, dtype=torch.float32),
            contributions=torch.tensor([contribution], device=self.device, dtype=torch.float32)
        )
        
        # 更新海马体中的记忆连接强度
        # BUG FIX: delta_w 是 shape [1] 的 tensor，但 update_memory_strength 期望 float。
        # 原代码直接传入 tensor，虽然 PyTorch 允许 float+tensor 运算，但语义不正确——
        # 应该取标量值（该 tensor 只有1个元素）。
        if hasattr(hippocampus_module, 'update_memory_strength'):
            hippocampus_module.update_memory_strength(memory_anchor_id, delta_w.item())


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
        self.eval_period = getattr(getattr(config, 'self_loop', None), 'mode3_eval_period', 10)  # 每 10 周期一次自评判
        
        # BUG FIX: 预留额外元学习组件注册表。
        # 接口层（BrainAIInterface）在初始化完 inner_thought_engine、
        # global_workspace、goal_system 后，通过此方法注入，
        # 使 apply_meta_learning() 中的状态转换矩阵、竞争权重、
        # 目标奖励权重都能接收 STDP 奖励信号。
        self._extra_meta_components: Dict[str, nn.Module] = {}
    
    def record_activation(self, type: str, id: Any, timestamp: float):
        self.full_link_stdp.record_activation(type, id, timestamp)
        
    def set_contribution(self, type: str, contribution: float):
        self.full_link_stdp.set_contribution(type, contribution)
        
    def register_meta_component(self, name: str, module: nn.Module):
        """
        [新增] 注册额外元学习组件到 STDP 元学习闭环。
        
        供 BrainAIInterface 在初始化 inner_thought_engine、
        global_workspace、goal_system 后调用，使这些组件
        能在 apply_meta_learning() 中接收 STDP 奖励信号。
        
        Args:
            name: 组件名称（如 'inner_thought_engine'、'global_workspace'、'goal_system'）
            module: 组件实例（不需要是 nn.Module，但建议传入）
        """
        self._extra_meta_components[name] = module

    def reset(self):
        """重置 STDP 引擎状态"""
        self.cycle_count = 0
        self.full_link_stdp.activation_times_tensor.fill_(-1e9)
        self.full_link_stdp.contribution_cache.clear()
    
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
                timestamp=timestamp,
                reward=outputs.get('evaluation_score', 1.0) / 100.0,
                is_tool_call=inputs.get('is_tool_call', False)
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
        # BUG FIX: 原代码检查 'memory_anchor' in inputs，但 refresh_engine 传入的键是 'memory_anchor_id'。
        # 导致 update_hippocampus_gate 从未被调用，海马体门控的 STDP 权重永远不更新。
        if 'hippocampus' in model_components and 'memory_anchor_id' in inputs:
            self.full_link_stdp.update_hippocampus_gate(
                memory_anchor_id=inputs.get('memory_anchor_id', 'unknown'),
                contribution=outputs.get('memory_contribution', 0.5),
                hippocampus_module=model_components['hippocampus'],
                timestamp=timestamp
            )
            
        # ========== 5. [动态化] 元学习闭环 (意识调度参数 STDP 更新) ==========
        if 'evaluation_score' in outputs:
            # BUG FIX: 原代码使用 (evaluation_score / 40.0) * 2 - 1 归一化，
            # 但 _apply_real_stdp_update() 传入的 evaluation_score = reward * 100，
            # 其中 reward 是 0-1 范围的置信度，所以 evaluation_score 范围为 0-100。
            # 用 /40 会导致 reward 超出 [-1, 1] 范围（如 reward=0.8 时算出 3.0）。
            # 改为使用 /100.0 使满分归一化到 1.0，再映射到 [-1, 1]：
            raw_score = outputs['evaluation_score']
            reward = (raw_score / 100.0) * 2 - 1  # 0→-1, 50→0, 100→1，clamp 防溢出
            reward = max(-1.0, min(1.0, reward))
            # BUG FIX: 将 inner_thought_engine、global_workspace、goal_system
            # 注入到 model_components，使 apply_meta_learning 中的相关分支生效。
            # 原代码 model_components 只包含 attention/ffn/hippocampus，
            # 导致 InnerThoughtEngine 状态转换矩阵和 GoalSystem 奖励权重
            # 从未接收到 STDP 信号。
            meta_components = dict(model_components)  # 浅拷贝，不污染原字典
            if hasattr(self, '_extra_meta_components'):
                meta_components.update(self._extra_meta_components)
            self.apply_meta_learning(
                reward=reward,
                model_components=meta_components
            )
    
    def apply_meta_learning(self, reward: float, model_components: Dict[str, nn.Module]):
        """
        [新增] 元学习闭环机制
        将 STDP 奖励信号传播到各个意识调度组件，更新非神经网络的动态规则参数：
        1. 状态转换矩阵概率 (InnerThoughtEngine)
        2. 全局工作空间竞争权重 (GlobalWorkspace)
        3. 目标系统的目标收益权重 (GoalSystem) -> 这一步在接口层处理
        """
        if not self.config.stdp.update_self_eval:
            return
            
        # 1. 更新 InnerThoughtEngine 的状态转换矩阵
        # BUG FIX: 原代码为 pass，STDP 奖励信号从未传递到独白引擎。
        # 现在通过 _transition_state(reward=...) 将奖励注入状态转换矩阵，
        # 使高频思维路径获得 LTP 增强，低效路径获得 LTD 削弱。
        if 'inner_thought_engine' in model_components:
            engine = model_components['inner_thought_engine']
            if hasattr(engine, '_transition_state') and callable(engine._transition_state):
                engine._transition_state(reward=reward)

        # 2. 更新 GlobalWorkspace 的竞争权重
        if 'global_workspace' in model_components:
            gw = model_components['global_workspace']
            if hasattr(gw, 'update_from_stdp_signal'):
                gw.update_from_stdp_signal(reward)

        # 3. 更新 GoalSystem 的目标奖励权重
        # BUG FIX: 原代码完全缺失 GoalSystem 的 STDP 反馈。
        # 正奖励（reward > 0）→ 当前目标类型权重上升（LTP）
        # 负奖励（reward < 0）→ 当前目标类型权重下降（LTD）
        if 'goal_system' in model_components:
            gs = model_components['goal_system']
            if hasattr(gs, 'current_goal') and gs.current_goal is not None:
                if reward > 0:
                    gs.update_reward_from_feedback(positive=True)
                elif reward < 0:
                    gs.update_reward_from_feedback(positive=False)
    
    def get_stats(self) -> dict:
        """获取 STDP 更新统计信息"""
        num_active = (self.full_link_stdp.activation_times_tensor > -1e8).sum().item()
        return {
            'cycle_count': self.cycle_count,
            'num_tracked_activations': num_active,
            'device': self.device
        }
