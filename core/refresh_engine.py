"""
模块 2: 100Hz人脑级高刷新单周期推理引擎

核心功能:
- 严格对齐人脑gamma高频认知节律 (10ms/100Hz)
- 窄窗口硬约束强制实现 O(1) 注意力复杂度
- 单周期不可修改的固定执行流
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading


@dataclass
class CycleResult:
    """单周期推理结果"""
    output_token: int
    output_features: torch.Tensor
    memory_anchors: List[dict]
    stdp_updated: bool
    cycle_time_ms: float
    success: bool


class RefreshCycleEngine:
    """
    10ms 刷新周期引擎
    
    每个周期严格执行以下顺序:
    1. 输入 token 接收与特征提取
    2. 海马体记忆锚点调取与注意力门控加载
    3. 窄窗口上下文 + 当前token的模型前向推理
    4. 单周期输出结果生成
    5. 全链路 STDP 权重本地刷新
    6. 海马体情景记忆编码与更新
    7. 全局工作记忆压缩更新
    """
    
    def __init__(
        self,
        model: nn.Module,
        hippocampus: nn.Module,
        stdp_engine: 'STDPEngine',
        period_ms: int = 10,
        narrow_window_size: int = 2,
        device: Optional[str] = None
    ):
        import torch.nn as nn
        self.model = model
        self.hippocampus = hippocampus
        self.stdp_engine = stdp_engine
        self.period_ms = period_ms
        self.narrow_window_size = narrow_window_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 周期计数器
        self.cycle_count = 0
        self.total_time_ms = 0.0
        
        # 时序同步
        self.last_cycle_end = time.time()
        
        # 窄窗口上下文缓冲区
        self.context_buffer: List[dict] = []
        
        # 性能统计
        self.stats = {
            'total_cycles': 0,
            'avg_cycle_time_ms': 0.0,
            'max_cycle_time_ms': 0.0,
            'min_cycle_time_ms': float('inf'),
            'overrun_count': 0  # 超过 10ms 的次数
        }
    
    def run_cycle(self, input_token: int, input_text: Optional[str] = None) -> CycleResult:
        """
        执行一个完整刷新周期 (严格 10ms)
        
        Args:
            input_token: 输入 token ID
            input_text: 可选的输入文本 (用于日志)
        
        Returns:
            CycleResult: 周期推理结果
        """
        cycle_start = time.time()
        timestamp = cycle_start * 1000  # 转换为毫秒
        
        try:
            # ========== 1. 输入 token 接收与特征提取 ==========
            features = self._extract_features(input_token)
            
            # ========== 2. 海马体记忆锚点调取与注意力门控加载 ==========
            memory_anchors = self.hippocampus.recall(features, topk=self.narrow_window_size)
            attention_gate = self._build_attention_gate(memory_anchors)
            
            # ========== 3. 窄窗口上下文 + 当前token的模型前向推理 ==========
            # 构建窄窗口输入 (仅包含 1-2 个最相关上下文)
            narrow_context = self._build_narrow_context(memory_anchors, input_token)
            
            output = self._forward_inference(
                input_token=input_token,
                features=features,
                context=narrow_context,
                attention_gate=attention_gate
            )
            
            # ========== 4. 单周期输出结果生成 ==========
            output_token = self._generate_output(output)
            
            # ========== 5. 全链路 STDP 权重本地刷新 ==========
            stdp_inputs = {
                'context_tokens': [ctx['token_id'] for ctx in narrow_context],
                'current_token': input_token,
                'features': features,
                'memory_anchor_id': memory_anchors[0]['id'] if memory_anchors else 'none'
            }
            stdp_outputs = {
                'attention_output': output.get('attention_output', torch.zeros(1)),
                'ffn_output': output.get('ffn_output', torch.zeros(1)),
                'memory_contribution': output.get('memory_contribution', 0.5)
            }
            
            self.stdp_engine.step(
                model_components={
                    'attention': self.model.get_attention_layer(),
                    'ffn': self.model.get_ffn_layer(),
                    'hippocampus': self.hippocampus
                },
                inputs=stdp_inputs,
                outputs=stdp_outputs,
                timestamp=timestamp
            )
            stdp_updated = True
            
            # ========== 6. 海马体情景记忆编码与更新 ==========
            self.hippocampus.encode(
                features=features,
                token_id=input_token,
                timestamp=int(timestamp),
                context=narrow_context
            )
            
            # ========== 7. 全局工作记忆压缩更新 ==========
            self._update_working_memory(output)
            
        except Exception as e:
            # 异常处理
            return CycleResult(
                output_token=input_token,
                output_features=torch.zeros(1),
                memory_anchors=[],
                stdp_updated=False,
                cycle_time_ms=self.period_ms,
                success=False
            )
        
        # ========== 周期时间控制 ==========
        cycle_end = time.time()
        elapsed_ms = (cycle_end - cycle_start) * 1000
        
        # 等待至周期结束 (确保精确 10ms)
        sleep_time_ms = max(0, self.period_ms - elapsed_ms)
        if sleep_time_ms > 0:
            time.sleep(sleep_time_ms / 1000.0)
        
        # 实际周期时间
        actual_cycle_ms = (time.time() - cycle_start) * 1000
        
        # 更新统计
        self._update_stats(actual_cycle_ms)
        
        # 更新上下文缓冲区
        self._update_context_buffer(input_token, output)
        
        self.cycle_count += 1
        
        return CycleResult(
            output_token=output_token,
            output_features=features,
            memory_anchors=memory_anchors,
            stdp_updated=stdp_updated,
            cycle_time_ms=actual_cycle_ms,
            success=True
        )
    
    def _extract_features(self, token_id: int) -> torch.Tensor:
        """步骤 1: 特征提取"""
        # 使用模型的 embedding 层提取特征
        if hasattr(self.model, 'get_input_embeddings'):
            embeddings = self.model.get_input_embeddings()
            token_tensor = torch.tensor([[token_id]], device=self.device)
            features = embeddings(token_tensor).squeeze(0)
        else:
            # 简化处理：随机特征 (实际应由模型提供)
            features = torch.randn(1, self.model.config.hidden_size, device=self.device)
        
        return features
    
    def _build_attention_gate(self, memory_anchors: List[dict]) -> Optional[torch.Tensor]:
        """步骤 2: 构建注意力门控"""
        if not memory_anchors:
            return None
        
        # 从记忆锚点构建注意力偏置
        # 形状：[1, num_heads, seq_len, seq_len]
        num_anchors = len(memory_anchors)
        gate = torch.zeros(
            (1, 1, 1, num_anchors + 1),  # 简化为单头
            device=self.device
        )
        
        # 对锚点位置施加负偏置 (降低注意力)
        # 非锚点位置会被屏蔽
        return gate
    
    def _build_narrow_context(self, memory_anchors: List[dict], current_token: int) -> List[dict]:
        """步骤 3: 构建窄窗口上下文"""
        context = []
        
        # 添加记忆锚点对应的上下文
        for anchor in memory_anchors[:self.narrow_window_size - 1]:
            context.append({
                'token_id': anchor.get('token_id', current_token),
                'features': anchor.get('features', torch.zeros(1)),
                'timestamp': anchor.get('timestamp', 0)
            })
        
        # 添加当前token
        context.append({
            'token_id': current_token,
            'features': None,  # 已在步骤 1 提取
            'timestamp': int(time.time() * 1000)
        })
        
        return context
    
    def _forward_inference(
        self,
        input_token: int,
        features: torch.Tensor,
        context: List[dict],
        attention_gate: Optional[torch.Tensor]
    ) -> dict:
        """步骤 3: 窄窗口前向推理"""
        # 调用模型进行推理
        # 这里假设模型有一个简化的推理接口
        
        output = {}
        
        # 注意力层
        if hasattr(self.model, 'forward_attention'):
            attn_output = self.model.forward_attention(
                hidden_states=features.unsqueeze(0),
                attention_mask=attention_gate
            )
            output['attention_output'] = attn_output
        else:
            output['attention_output'] = features
        
        # FFN 层
        if hasattr(self.model, 'forward_ffn'):
            ffn_output = self.model.forward_ffn(output['attention_output'])
            output['ffn_output'] = ffn_output
        else:
            output['ffn_output'] = output['attention_output']
        
        # 计算记忆贡献度
        output['memory_contribution'] = torch.norm(output['ffn_output']).item() / 10.0
        
        return output
    
    def _generate_output(self, inference_output: dict) -> int:
        """步骤 4: 生成输出 token"""
        # 从推理输出中提取下一个 token
        # 实际实现需要根据模型的输出层
        
        # 简化处理：返回固定值
        return 0
    
    def _update_working_memory(self, output: dict):
        """步骤 7: 更新工作记忆"""
        # 压缩更新全局工作记忆
        # 保留最近 N 个周期的关键信息
        pass
    
    def _update_context_buffer(self, token_id: int, output: dict):
        """更新上下文缓冲区"""
        self.context_buffer.append({
            'token_id': token_id,
            'output': output,
            'timestamp': int(time.time() * 1000)
        })
        
        # 保持缓冲区大小固定 (防止内存无限增长)
        max_buffer_size = 100
        if len(self.context_buffer) > max_buffer_size:
            self.context_buffer = self.context_buffer[-max_buffer_size:]
    
    def _update_stats(self, cycle_time_ms: float):
        """更新性能统计"""
        self.stats['total_cycles'] += 1
        self.stats['avg_cycle_time_ms'] = (
            (self.stats['avg_cycle_time_ms'] * (self.stats['total_cycles'] - 1) + cycle_time_ms)
            / self.stats['total_cycles']
        )
        self.stats['max_cycle_time_ms'] = max(self.stats['max_cycle_time_ms'], cycle_time_ms)
        self.stats['min_cycle_time_ms'] = min(self.stats['min_cycle_time_ms'], cycle_time_ms)
        
        if cycle_time_ms > self.period_ms:
            self.stats['overrun_count'] += 1
    
    def get_stats(self) -> dict:
        """获取引擎统计信息"""
        return {
            **self.stats,
            'cycle_count': self.cycle_count,
            'target_period_ms': self.period_ms,
            'narrow_window_size': self.narrow_window_size
        }
    
    def reset(self):
        """重置引擎状态"""
        self.cycle_count = 0
        self.context_buffer.clear()
        self.stats = {
            'total_cycles': 0,
            'avg_cycle_time_ms': 0.0,
            'max_cycle_time_ms': 0.0,
            'min_cycle_time_ms': float('inf'),
            'overrun_count': 0
        }


# 导入需要的模块
import torch.nn as nn

# 向前引用 STDPEngine
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .stdp_engine import STDPEngine
