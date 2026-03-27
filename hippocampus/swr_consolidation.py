"""
SWR (Sharp Wave Ripple) - 离线回放巩固单元

功能:
- 端侧空闲时，模拟人脑睡眠尖波涟漪
- 回放记忆序列与推理过程
- 完成记忆巩固、权重优化、记忆修剪
- 不占用推理算力
"""

import torch
import torch.nn as nn
import time
import threading
import math
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import random


@dataclass
class ReplaySequence:
    """回放序列数据"""
    sequence_id: str
    memories: List[dict]
    reward_signal: float  # 奖励信号 (来自自评判)
    timestamp: int


class SWRConsolidation:
    """
    尖波涟漪离线回放巩固
    
    模拟人脑睡眠期间的记忆巩固机制:
    1. 在设备空闲时触发
    2. 回放近期的记忆序列和推理过程
    3. 通过 STDP 规则强化正确路径，修剪错误路径
    4. 自动遗忘无效记忆
    """
    
    def __init__(
        self,
        config,
        hippocampus_module: Optional[nn.Module] = None,
        idle_threshold_s: int = 300,      # 空闲 5 分钟触发
        replay_frequency: float = 0.1,     # 回放频率 (Hz)
        max_replay_sequences: int = 10,    # 最大回放序列数
        consolidation_strength: float = 0.5  # 巩固强度
    ):
        self.config = config
        self.hippocampus = hippocampus_module
        self.idle_threshold_s = idle_threshold_s
        self.replay_frequency = replay_frequency
        self.max_replay_sequences = max_replay_sequences
        self.consolidation_strength = consolidation_strength
        
        # ========== 回放缓冲区 ==========
        self.replay_buffer: List[ReplaySequence] = []
        
        # ========== 状态标志 ==========
        self.is_idle = False
        self.last_activity_time = time.time()
        self.consolidation_thread: Optional[threading.Thread] = None
        self.stop_flag = False
        
        # ========== 回调函数 ==========
        self.stdp_update_fn: Optional[Callable] = None
        self.memory_prune_fn: Optional[Callable] = None
    
    def start_monitoring(self):
        """开始监控设备空闲状态"""
        self.stop_flag = False
        self.consolidation_thread = threading.Thread(
            target=self._idle_monitor_loop,
            daemon=True
        )
        self.consolidation_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.stop_flag = True
        if self.consolidation_thread:
            self.consolidation_thread.join(timeout=5.0)
    
    def record_activity(self):
        """记录用户活动 (重置空闲计时器)"""
        self.last_activity_time = time.time()
        self.is_idle = False
    
    def add_replay_sequence(
        self,
        sequence_id: str,
        memories: List[dict],
        reward_signal: float
    ):
        """
        添加回放序列到缓冲区
        
        Args:
            sequence_id: 序列 ID
            memories: 记忆序列
            reward_signal: 奖励信号 (0-1, 越高表示越值得巩固)
        """
        sequence = ReplaySequence(
            sequence_id=sequence_id,
            memories=memories,
            reward_signal=reward_signal,
            timestamp=int(time.time() * 1000)
        )
        
        self.replay_buffer.append(sequence)
        
        # 保持缓冲区大小固定
        if len(self.replay_buffer) > self.max_replay_sequences:
            # 删除奖励信号最低的序列
            self.replay_buffer.sort(key=lambda s: s.reward_signal, reverse=True)
            self.replay_buffer = self.replay_buffer[:self.max_replay_sequences]
    
    def _idle_monitor_loop(self):
        """空闲监控循环 (后台线程)"""
        while not self.stop_flag:
            current_time = time.time()
            idle_duration = current_time - self.last_activity_time
            
            # 检查是否进入空闲状态
            if idle_duration >= self.idle_threshold_s and not self.is_idle:
                self.is_idle = True
                
                # 触发离线回放巩固
                if self.replay_buffer:
                    self._run_consolidation()
            
            # 定期检查 (每 10 秒)
            time.sleep(10.0)
    
    def _run_consolidation(self):
        """执行离线回放巩固"""
        if not self.hippocampus or not self.replay_buffer:
            return
        
        # logger.debug(f"[SWR] 开始离线回放巩固，缓冲区序列数：{len(self.replay_buffer)}")
        
        # ========== 1. 按奖励信号排序，优先回放高奖励序列 ==========
        sorted_sequences = sorted(
            self.replay_buffer, 
            key=lambda s: s.reward_signal, 
            reverse=True
        )
        
        # ========== 2. 回放每个序列 ==========
        for seq in sorted_sequences:
            self._replay_sequence(seq)
            
            # 模拟尖波涟漪的短暂间隔
            time.sleep(1.0 / self.replay_frequency)
        
        # ========== 3. 记忆修剪 ==========
        if self.memory_prune_fn and self.hippocampus:
            pruned_count = self.memory_prune_fn(threshold=0.3)
            # logger.debug(f"[SWR] 修剪了 {pruned_count} 个弱记忆")
        
        # ========== 4. 清空回放缓冲区 ==========
        self.replay_buffer.clear()
        
        # logger.debug("[SWR] 离线回放巩固完成")
    
    def _replay_sequence(self, sequence: ReplaySequence):
        """回放单个序列"""
        # ========== 模拟海马体 sharp wave ==========
        # 快速重放记忆序列
        for i, memory in enumerate(sequence.memories):
            # 增强记忆的激活强度 (巩固)
            # ca3_memory 是 HippocampusSystem 的必需组件
            memory_id = memory.get('memory_id')
            if memory_id:
                # 根据奖励信号增强
                delta = sequence.reward_signal * self.consolidation_strength
                self.hippocampus.ca3_memory.update_memory_strength(memory_id, delta)
            
            # 调用 STDP 更新 (如果已设置回调)
            if self.stdp_update_fn:
                self.stdp_update_fn(memory, sequence.reward_signal)
        
        # ========== 模拟 ripple 振荡 ==========
        # 在 replay 结束后施加高频振荡
        self._apply_ripple_oscillation(sequence)
    
    def _apply_ripple_oscillation(self, sequence: ReplaySequence):
        """
        施加 ripple 振荡（生产级实现）
        
        模拟生物海马体的 Sharp Wave Ripple (SWR) 振荡：
        - 频率：150-250 Hz（模拟生物节律）
        - 特点：高频、短时、强同步
        - 作用：记忆序列的时间压缩回放和突触巩固
        
        实现原理：
        1. 相位调制：模拟正弦振荡的相位变化
        2. 强度渐变：ripple 开始和结束强度较低，中间最高
        3. 时间压缩：将长时记忆序列压缩到短时高频回放
        4. 序列同步：相邻记忆之间的相位同步增强
        """
        # 生物节律参数
        base_frequency = 200  # Hz (150-250 Hz 的中值)
        ripple_duration_ms = 100  # 每次 ripple 持续约 100ms
        num_cycles = int(base_frequency * ripple_duration_ms / 1000)  # 约 20 个周期
        
        # 模拟时间压缩：每个记忆对应一个振荡周期
        num_memories = len(sequence.memories)
        if num_memories == 0:
            return
        
        # 记忆间的时间间隔（压缩后的）
        cycle_duration_ms = ripple_duration_ms / num_memories
        
        # 基础巩固强度（来自序列的奖励信号）
        base_consolidation = self.consolidation_strength * sequence.reward_signal
        
        # ========== 1. 相位调制循环 ==========
        for cycle_idx, memory in enumerate(sequence.memories):
            # 计算当前周期相位 (0 到 2π)
            phase = 2 * 3.14159 * cycle_idx / num_cycles
            
            # 正弦调制：相位决定当前的"兴奋度"
            # 在波峰时（phase = π/2, 5π/2, ...）强度最高
            phase_modulation = 0.5 + 0.5 * math.sin(phase)
            
            # ========== 2. 强度渐变 ==========
            # Ripple 的强度曲线：开始低 -> 中间高 -> 结束低
            # 使用高斯曲线模拟
            progress = cycle_idx / max(num_memories - 1, 1)  # 0 到 1
            envelope = math.exp(-((progress - 0.5) ** 2) / 0.1)  # 高斯包络
            
            # 综合强度
            oscillation_strength = base_consolidation * phase_modulation * envelope
            
            # ========== 3. 应用到记忆 ==========
            memory_id = memory.get('memory_id')
            if memory_id and hasattr(self.hippocampus, 'ca3_memory'):
                # 基础巩固
                self.hippocampus.ca3_memory.update_memory_strength(
                    memory_id, 
                    oscillation_strength
                )
                
                # ========== 4. 序列同步增强 ==========
                # 相邻记忆之间的连接加强
                if cycle_idx > 0:
                    prev_memory_id = sequence.memories[cycle_idx - 1].get('memory_id')
                    if prev_memory_id:
                        # 增强前后记忆的关联（模拟时间序列学习）
                        self._strengthen_memory_link(
                            prev_memory_id, 
                            memory_id, 
                            oscillation_strength * 0.5
                        )
            
            # ========== 5. 突触增强噪声 ==========
            # 添加适度的随机性，模拟生物噪声
            noise = random.gauss(0, 0.02) * oscillation_strength
            if memory_id and hasattr(self.hippocampus, 'ca3_memory'):
                self.hippocampus.ca3_memory.update_memory_strength(
                    memory_id, 
                    noise
                )
        
        # ========== 6. Ripple 后的快速抑制 ==========
        # 模拟生物 SWR 后的短暂抑制期
        # 这有助于防止过度激活和保持记忆的选择性
        self._apply_post_ripple_inhibition(sequence)
    
    def _strengthen_memory_link(
        self, 
        from_memory_id: str, 
        to_memory_id: str, 
        strength: float
    ):
        """
        增强两个记忆之间的关联
        
        模拟 CA3-CA1 的序列学习：
        - 前一个记忆的激活促进后一个记忆的激活
        - 这种时序依赖性是情景记忆的关键
        """
        if not hasattr(self, '_memory_links'):
            self._memory_links = {}  # 存储记忆关联
        
        link_key = f"{from_memory_id}->{to_memory_id}"
        
        if link_key in self._memory_links:
            # 累积增强
            self._memory_links[link_key] += strength
            # 饱和限制
            self._memory_links[link_key] = min(1.0, self._memory_links[link_key])
        else:
            self._memory_links[link_key] = strength
        
        # 如果海马体支持关联更新 - update_link 是 CA3EpisodicMemory 的方法
        if hasattr(self.hippocampus.ca3_memory, 'update_link'):
            self.hippocampus.ca3_memory.update_link(
                from_memory_id, 
                to_memory_id, 
                self._memory_links[link_key]
            )
    
    def _apply_post_ripple_inhibition(self, sequence: ReplaySequence):
        """
        Ripple 后的快速抑制
        
        生物原理：SWR 后有一个短暂的抑制期，
        这有助于：
        1. 防止过度激活
        2. 保持记忆的选择性
        3. 为下一轮 SWR 做准备
        """
        # 抑制强度随时间衰减
        inhibition_decay = 0.95
        
        # 对序列中的记忆应用轻微抑制
        # 这实际上是选择性地"修剪"弱记忆
        for memory in sequence.memories:
            memory_id = memory.get('memory_id')
            if memory_id:
                # 获取当前记忆强度
                current_strength = self.hippocampus.ca3_memory.get_memory_strength(memory_id)
                
                # 如果记忆强度过低，应用更强的抑制（修剪弱记忆）
                if current_strength < 0.2:
                    inhibition = -0.05  # 更强的抑制
                else:
                    inhibition = -0.01 * inhibition_decay  # 轻微抑制
                
                self.hippocampus.ca3_memory.update_memory_strength(memory_id, inhibition)
    
    def set_callbacks(
        self,
        stdp_update_fn: Optional[Callable] = None,
        memory_prune_fn: Optional[Callable] = None
    ):
        """设置回调函数"""
        self.stdp_update_fn = stdp_update_fn
        self.memory_prune_fn = memory_prune_fn
    
    def get_stats(self) -> dict:
        """获取 SWR 统计信息"""
        return {
            'is_idle': self.is_idle,
            'idle_duration_s': time.time() - self.last_activity_time,
            'replay_buffer_size': len(self.replay_buffer),
            'avg_reward_signal': (
                sum(s.reward_signal for s in self.replay_buffer) / len(self.replay_buffer)
                if self.replay_buffer else 0.0
            ),
            'consolidation_thread_alive': (
                self.consolidation_thread.is_alive() 
                if self.consolidation_thread else False
            )
        }
    
    def trigger_manual_consolidation(self):
        """手动触发回放巩固 (用于测试)"""
        if self.replay_buffer:
            self._run_consolidation()
        else:
            # logger.debug("[SWR] 回放缓冲区为空，无法触发巩固")
            pass
