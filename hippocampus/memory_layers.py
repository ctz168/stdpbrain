"""
记忆分层系统 - Memory Tiering System

模拟人脑记忆的层次化结构:
- 短期记忆 (Short-Term): 工作记忆，容量有限，快速衰减（秒~分钟级）
- 中期记忆 (Mid-Term): 近期事件，适度衰减（小时~天级）
- 长期记忆 (Long-Term): 重要固化记忆，极慢衰减（月~年级）

固化规则:
- 短期 → 中期: 被召回 ≥ 2 次 或 生存时间 > 30 分钟 且 激活强度 > 0.5
- 中期 → 长期: 被召回 ≥ 5 次 或 生存时间 > 24 小时 且 激活强度 > 1.0
- 降级规则: 长期记忆如果连续 10 次未被召回且激活强度 < 0.3, 降级为中期
"""

import time
import enum
from typing import Optional
from dataclasses import dataclass, field


class MemoryTier(enum.IntEnum):
    """记忆层级"""
    SHORT_TERM = 0   # 短期记忆
    MID_TERM = 1     # 中期记忆
    LONG_TERM = 2    # 长期记忆


@dataclass
class TierConfig:
    """分层配置参数"""
    # 短期记忆配置
    short_term_decay_rate: float = 0.990       # 每次衰减后的保留比例（快速衰减）
    short_term_max_capacity: int = 5000        # 短期记忆最大容量
    short_term_recall_threshold: float = 0.5   # 短期记忆召回阈值
    
    # 中期记忆配置
    mid_term_decay_rate: float = 0.998         # 中期衰减（较慢）
    mid_term_max_capacity: int = 20000         # 中期记忆最大容量
    mid_term_recall_threshold: float = 0.4     # 中期记忆召回阈值
    
    # 长期记忆配置
    long_term_decay_rate: float = 0.9999       # 长期衰减（极慢）
    long_term_max_capacity: int = 30000        # 长期记忆最大容量
    long_term_recall_threshold: float = 0.3    # 长期记忆召回阈值
    
    # 固化规则
    promote_stm_to_mtm_min_recalls: int = 2    # 短期→中期: 最小被召回次数
    promote_stm_to_mtm_min_strength: float = 0.5  # 短期→中期: 最小激活强度
    promote_stm_to_mtm_min_age_s: float = 1800.0  # 短期→中期: 最小存在时间(秒, 30分钟)
    
    promote_mtm_to_ltm_min_recalls: int = 5    # 中期→长期: 最小被召回次数
    promote_mtm_to_ltm_min_strength: float = 1.0  # 中期→长期: 最小激活强度
    promote_mtm_to_ltm_min_age_s: float = 86400.0  # 中期→长期: 最小存在时间(秒, 24小时)
    
    # 降级规则
    demote_ltm_min_misses: int = 10            # 长期→中期: 连续未被召回次数
    demote_ltm_max_strength: float = 0.3       # 长期→中期: 最大激活强度
    
    demote_mtm_min_misses: int = 15            # 中期→短期: 连续未被召回次数
    demote_mtm_max_strength: float = 0.2       # 中期→短期: 最大激活强度


class MemoryConsolidationManager:
    """
    记忆固化/降级管理器
    
    模拟人脑的记忆固化机制:
    - 海马体中短期记忆 → 经过反复回放(SWR) → 固化为长期记忆(新皮层)
    - 不常用的长期记忆逐渐弱化
    """
    
    def __init__(self, config: Optional[TierConfig] = None):
        self.config = config or TierConfig()
    
    def get_decay_rate(self, tier: MemoryTier) -> float:
        """获取对应层级的衰减率"""
        if tier == MemoryTier.SHORT_TERM:
            return self.config.short_term_decay_rate
        elif tier == MemoryTier.MID_TERM:
            return self.config.mid_term_decay_rate
        else:
            return self.config.long_term_decay_rate
    
    def get_recall_threshold(self, tier: MemoryTier) -> float:
        """获取对应层级的召回阈值"""
        if tier == MemoryTier.SHORT_TERM:
            return self.config.short_term_recall_threshold
        elif tier == MemoryTier.MID_TERM:
            return self.config.mid_term_recall_threshold
        else:
            return self.config.long_term_recall_threshold
    
    def get_max_capacity(self, tier: MemoryTier) -> int:
        """获取对应层级的最大容量"""
        if tier == MemoryTier.SHORT_TERM:
            return self.config.short_term_max_capacity
        elif tier == MemoryTier.MID_TERM:
            return self.config.mid_term_max_capacity
        else:
            return self.config.long_term_max_capacity
    
    def should_promote(self, memory) -> Optional[MemoryTier]:
        """
        判断记忆是否应该被提升到更高层级
        
        Args:
            memory: EpisodicMemory 对象，必须具有以下属性:
                - tier: MemoryTier
                - recall_count: int (被召回次数)
                - activation_strength: float
                - timestamp: int (创建时间戳, 毫秒)
                - consecutive_misses: int (连续未被召回次数)
        
        Returns:
            如果应该提升，返回目标层级；否则返回 None
        """
        current_tier = getattr(memory, 'tier', MemoryTier.SHORT_TERM)
        recall_count = getattr(memory, 'recall_count', 0)
        strength = getattr(memory, 'activation_strength', 0.0)
        age_s = (time.time() * 1000 - memory.timestamp) / 1000.0  # 秒
        
        if current_tier == MemoryTier.SHORT_TERM:
            # 短期 → 中期: 被召回≥2次 或 (存在>30分钟 且 强度>0.5)
            if recall_count >= self.config.promote_stm_to_mtm_min_recalls:
                return MemoryTier.MID_TERM
            if (age_s >= self.config.promote_stm_to_mtm_min_age_s and 
                strength >= self.config.promote_stm_to_mtm_min_strength):
                return MemoryTier.MID_TERM
        
        elif current_tier == MemoryTier.MID_TERM:
            # 中期 → 长期: 被召回≥5次 或 (存在>24小时 且 强度>1.0)
            if recall_count >= self.config.promote_mtm_to_ltm_min_recalls:
                return MemoryTier.LONG_TERM
            if (age_s >= self.config.promote_mtm_to_ltm_min_age_s and 
                strength >= self.config.promote_mtm_to_ltm_min_strength):
                return MemoryTier.LONG_TERM
        
        return None
    
    def should_demote(self, memory) -> Optional[MemoryTier]:
        """
        判断记忆是否应该被降级到更低层级
        
        Args:
            memory: EpisodicMemory 对象
        
        Returns:
            如果应该降级，返回目标层级；否则返回 None
        """
        current_tier = getattr(memory, 'tier', MemoryTier.SHORT_TERM)
        consecutive_misses = getattr(memory, 'consecutive_misses', 0)
        strength = getattr(memory, 'activation_strength', 0.0)
        
        if current_tier == MemoryTier.LONG_TERM:
            # 长期 → 中期: 连续10次未被召回 且 激活强度<0.3
            if (consecutive_misses >= self.config.demote_ltm_min_misses and 
                strength < self.config.demote_ltm_max_strength):
                return MemoryTier.MID_TERM
        
        elif current_tier == MemoryTier.MID_TERM:
            # 中期 → 短期: 连续15次未被召回 且 激活强度<0.2
            if (consecutive_misses >= self.config.demote_mtm_min_misses and 
                strength < self.config.demote_mtm_max_strength):
                return MemoryTier.SHORT_TERM
        
        return None
    
    def apply_decay(self, memory):
        """
        对记忆应用时间衰减
        
        根据记忆所在层级使用不同的衰减率:
        - 短期记忆: 快速衰减 (0.990)
        - 中期记忆: 中等衰减 (0.998)
        - 长期记忆: 极慢衰减 (0.9999)
        """
        tier = getattr(memory, 'tier', MemoryTier.SHORT_TERM)
        decay_rate = self.get_decay_rate(tier)
        
        # 应用衰减
        memory.activation_strength *= decay_rate
        
        # 确保不低于最低阈值
        memory.activation_strength = max(0.05, memory.activation_strength)
    
    def consolidate_memories(self, memories_dict) -> dict:
        """
        批量处理记忆的固化和降级
        
        Args:
            memories_dict: OrderedDict[str, EpisodicMemory]
        
        Returns:
            dict: 固化/降级统计 {'promoted': int, 'demoted': int, 'decayed': int}
        """
        stats = {'promoted': 0, 'demoted': 0, 'decayed': 0}
        
        for memory_id, memory in memories_dict.items():
            # 1. 检查是否应该提升
            new_tier = self.should_promote(memory)
            if new_tier is not None:
                old_tier = memory.tier
                memory.tier = new_tier
                # 提升时增强激活强度（模拟固化效应）
                memory.activation_strength = min(memory.activation_strength * 1.2, 2.0)
                stats['promoted'] += 1
                continue
            
            # 2. 检查是否应该降级
            new_tier = self.should_demote(memory)
            if new_tier is not None:
                memory.tier = new_tier
                # 降级时降低激活强度
                memory.activation_strength *= 0.8
                stats['demoted'] += 1
                continue
            
            # 3. 应用时间衰减
            self.apply_decay(memory)
            stats['decayed'] += 1
        
        return stats
    
    def get_tier_stats(self, memories_dict) -> dict:
        """获取各层级记忆统计"""
        tier_counts = {MemoryTier.SHORT_TERM: 0, MemoryTier.MID_TERM: 0, MemoryTier.LONG_TERM: 0}
        tier_avg_strength = {MemoryTier.SHORT_TERM: 0.0, MemoryTier.MID_TERM: 0.0, MemoryTier.LONG_TERM: 0.0}
        
        for memory in memories_dict.values():
            tier = getattr(memory, 'tier', MemoryTier.SHORT_TERM)
            tier_counts[tier] += 1
            tier_avg_strength[tier] += memory.activation_strength
        
        for tier in tier_counts:
            if tier_counts[tier] > 0:
                tier_avg_strength[tier] /= tier_counts[tier]
        
        return {
            'short_term_count': tier_counts[MemoryTier.SHORT_TERM],
            'mid_term_count': tier_counts[MemoryTier.MID_TERM],
            'long_term_count': tier_counts[MemoryTier.LONG_TERM],
            'short_term_avg_strength': tier_avg_strength[MemoryTier.SHORT_TERM],
            'mid_term_avg_strength': tier_avg_strength[MemoryTier.MID_TERM],
            'long_term_avg_strength': tier_avg_strength[MemoryTier.LONG_TERM],
        }
