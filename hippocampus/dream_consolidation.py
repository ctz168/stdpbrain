"""
梦境巩固系统 - Dream Consolidation System

模拟人脑睡眠期间的记忆巩固机制:
- NREM 阶段: 系统性地回放、固化、泛化记忆（记忆分层迁移）
- REM 阶段: 创造性重组、远距关联、情绪加工、恐惧消退

与现有 SWRConsolidation 的区别:
- SWR: 简单的序列回放 + STDP 权重强化，面向单次空闲事件
- Dream: 完整的睡眠周期模拟（NREM/REM交替），含记忆泛化、创造性重组、情绪加工

神经科学依据:
- NREM (Non-REM) 睡眠: 海马体→新皮层的"自上而下"信息转移，记忆泛化，模式补全
- REM (Rapid Eye Movement) 睡眠: 皮层→海马体→杏仁核的"自下而上"情绪加工，创造性关联
- 典型成年人睡眠周期: ~90分钟/周期，N1→N2→N3→N2→REM 交替，每夜约4-6个完整周期
"""

import time
import enum
import random
import math
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .memory_layers import MemoryTier, MemoryConsolidationManager, TierConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构定义
# ============================================================================


class SleepPhase(enum.Enum):
    """
    睡眠阶段枚举
    
    对应人脑真实的睡眠阶段划分:
    - NREM_LIGHT (N1/N2): 浅睡期，心率下降，体温降低，偶尔出现睡眠纺锤波
    - NREM_DEEP  (N3):    深睡期/慢波睡眠(SWS)，海马体 sharp-wave ripple 集中出现
    - REM:                 快速眼动期，生动梦境，情绪记忆加工的关键窗口
    """
    NREM_LIGHT = "nrem_light"   # 浅睡期
    NREM_DEEP = "nrem_deep"     # 深睡期（慢波睡眠）
    REM = "rem"                  # 快速眼动期


class DreamEventType(enum.Enum):
    """
    梦境事件类型枚举
    
    每种类型对应一种在睡眠期间发生的认知加工操作
    """
    REACTIVATE = "reactivate"          # 记忆重激活（NREM: 系统性回放）
    CONSOLIDATE = "consolidate"        # 记忆固化（NREM: 层级迁移）
    GENERALIZE = "generalize"          # 记忆泛化（NREM: 模式提取与schema生成）
    STABILIZE = "stabilize"            # 长期记忆稳定化（NREM: 强化已有LTM）
    COMBINE = "combine"                # 创造性重组（REM: 跨记忆融合）
    EXTINCT = "extinguish"             # 恐惧消退（REM: 情绪脱敏）
    ASSOCIATE = "associate"            # 远距关联（REM: 无关记忆连接）
    INSIGHT = "insight"                # 创意洞察（REM: 新知识生成）


@dataclass
class DreamEvent:
    """
    梦境事件 - 单次记忆加工操作记录
    
    记录睡眠期间对记忆执行的每一次操作，
    便于事后分析梦境中的"记忆活动"。
    
    Attributes:
        memory_ids: 参与加工的记忆 ID 列表（单个操作可能涉及多个记忆）
        phase: 发生时的睡眠阶段
        event_type: 操作类型（重激活/泛化/重组等）
        result: 操作结果的文字描述（如"记忆A已迁移至长期"）
        timestamp: 事件发生的系统时间戳（秒）
        importance: 事件重要性权重（影响日志级别）
    """
    memory_ids: List[str]
    phase: SleepPhase
    event_type: DreamEventType
    result: str = ""
    timestamp: float = 0.0
    importance: float = 0.5  # 0.0-1.0, 用于筛选重要事件

    def __post_init__(self):
        """确保时间戳默认为当前时间"""
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'memory_ids': self.memory_ids,
            'phase': self.phase.value,
            'event_type': self.event_type.value,
            'result': self.result,
            'timestamp': self.timestamp,
            'importance': self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DreamEvent':
        """从字典反序列化"""
        return cls(
            memory_ids=data.get('memory_ids', []),
            phase=SleepPhase(data.get('phase', 'nrem_light')),
            event_type=DreamEventType(data.get('event_type', 'reactivate')),
            result=data.get('result', ''),
            timestamp=data.get('timestamp', 0.0),
            importance=data.get('importance', 0.5),
        )


@dataclass
class DreamSequence:
    """
    梦境序列 - 一次完整睡眠周期的记忆加工记录
    
    一次"梦境"由多个 DreamEvent 组成，类似于一场梦中的多个场景。
    每场梦境可能包含记忆回放、创造性跳跃、情绪加工等多个"场景"。
    
    Attributes:
        events: 梦境中的所有事件列表（按时间顺序）
        total_cycles: 经过的完整睡眠周期数
        memories_processed: 本场梦境中处理的记忆总数
        new_associations: 新建立的远距关联数
        creative_insights: 产生的创造性洞察列表（文字描述）
        phase_distribution: 各睡眠阶段的事件数统计
    """
    events: List[DreamEvent] = field(default_factory=list)
    total_cycles: int = 0
    memories_processed: int = 0
    new_associations: int = 0
    creative_insights: List[str] = field(default_factory=list)
    phase_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'events': [e.to_dict() for e in self.events],
            'total_cycles': self.total_cycles,
            'memories_processed': self.memories_processed,
            'new_associations': self.new_associations,
            'creative_insights': self.creative_insights,
            'phase_distribution': self.phase_distribution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DreamSequence':
        """从字典反序列化"""
        events = [DreamEvent.from_dict(e) for e in data.get('events', [])]
        return cls(
            events=events,
            total_cycles=data.get('total_cycles', 0),
            memories_processed=data.get('memories_processed', 0),
            new_associations=data.get('new_associations', 0),
            creative_insights=data.get('creative_insights', []),
            phase_distribution=data.get('phase_distribution', {}),
        )

    def get_summary(self) -> str:
        """生成梦境摘要"""
        lines = [
            f"梦境报告: 共 {self.total_cycles} 个周期",
            f"  处理记忆: {self.memories_processed} 条",
            f"  新建关联: {self.new_associations} 条",
            f"  创意洞察: {len(self.creative_insights)} 条",
        ]
        if self.creative_insights:
            lines.append("  洞察详情:")
            for insight in self.creative_insights[:5]:
                lines.append(f"    - {insight}")

        phase_str = " → ".join(
            f"{k}:{v}" for k, v in self.phase_distribution.items()
        )
        lines.append(f"  阶段分布: {phase_str}")

        return "\n".join(lines)


@dataclass
class DreamConfig:
    """
    梦境巩固系统的配置参数
    
    所有参数都基于神经科学研究的近似值，经过工程化调整以适应计算环境。
    """
    # --- 睡眠调度 ---
    idle_trigger_minutes: float = 30.0          # 空闲超过30分钟自动触发睡眠
    typical_sleep_cycles: int = 4               # 典型睡眠周期数（NREM→REM × 4）
    sleep_cycle_duration_s: float = 90.0        # 每个周期约90分钟（人类标准）

    # --- NREM 参数 ---
    nrem_max_memories_per_cycle: int = 50       # 每次NREM周期最多处理50条记忆
    nrem_generalization_threshold: int = 3      # 至少3条相似记忆才触发泛化
    nrem_pattern_similarity_threshold: float = 0.6  # 模式相似度阈值（用于泛化检测）
    nrem_stabilization_boost: float = 1.05      # 长期记忆稳定化强度提升系数

    # --- REM 参数 ---
    rem_max_recombinations: int = 5             # 每次REM周期最多生成5个重组记忆
    rem_distant_association_max: int = 3        # 每次REM周期最多建立3个远距关联
    rem_fear_extinction_rate: float = 0.95      # 恐惧消退率（每REM周期 emotion *= 0.95）
    rem_fear_minimum_intensity: float = 0.1     # 恐惧消退最低保留强度（生存相关恐惧）
    rem_creative_jump_probability: float = 0.3  # 创造性跳跃概率（梦境中的随机转移）
    rem_surreal_blend_probability: float = 0.15 # 超现实融合概率（不同记忆元素的混合）

    # --- 效率限制 ---
    max_total_memories_per_sleep: int = 200     # 整个睡眠周期最多处理200条记忆
    dream_sequence_max_events: int = 500        # 梦境序列最大事件数
    creative_insight_max_length: int = 5        # 洞察描述最大条数

    # --- 情绪处理 ---
    positive_emotion_consolidation_rate: float = 1.02   # 正面情绪增强率
    negative_emotion_decay_rate: float = 0.98          # 负面情绪衰减率
    survival_related_emotions: List[str] = field(
        default_factory=lambda: ["fear", "danger", "pain", "threat", "恐惧", "危险"]
    )  # 生存相关情绪类型（消退时保留最低强度）


# ============================================================================
# 梦境巩固系统核心类
# ============================================================================


class DreamConsolidationSystem:
    """
    梦境巩固系统 - 完整的睡眠记忆加工模拟器
    
    模拟人脑在睡眠期间的两阶段记忆加工:
    
    1. NREM 阶段（浅睡→深睡）:
       - 按重要性排序系统性地回放短期记忆
       - 将符合条件的记忆从短期→中期→长期迁移
       - 提取相似记忆中的共同模式，生成"schema"（泛化记忆）
       - 稳定化现有长期记忆（增强激活强度）
    
    2. REM 阶段（快速眼动）:
       - 创造性重组: 跨记忆提取元素进行融合
       - 远距关联: 在看似无关的记忆间建立联系
       - 情绪加工: 正面情绪增强，负面情绪逐渐消退
       - 恐惧消退: 系统性降低恐惧类记忆的情绪强度
       - 创意洞察: 从重组中生成新的知识关联
    
    使用方式:
        system = DreamConsolidationSystem()
        # 注册记忆源（CA3情景记忆库）
        system.register_memory_source(ca3_memory)
        # 启动睡眠监控
        system.start_monitoring()
        # 或手动触发一次完整睡眠
        sequence = system.run_full_sleep_cycle()
        print(sequence.get_summary())
    
    序列化:
        state = system.get_state()
        system.set_state(state)
    """

    def __init__(self, config: Optional[DreamConfig] = None):
        """
        初始化梦境巩固系统
        
        Args:
            config: 梦境系统配置参数，为 None 时使用默认配置
        """
        self.config = config or DreamConfig()

        # ========== 记忆源（外部注入）==========
        # CA3 情景记忆库的引用，用于读取和修改记忆
        self._memory_source: Optional[Any] = None  # CA3EpisodicMemory 实例

        # ========== 睡眠状态机 ==========
        self._is_sleeping: bool = False
        self._current_phase: Optional[SleepPhase] = None  # None 表示清醒状态
        self._current_cycle: int = 0          # 当前周期编号（从0开始）
        self._last_activity_time: float = time.time()

        # ========== 梦境历史记录 ==========
        self._dream_sequences: List[DreamSequence] = []
        self._current_dream: Optional[DreamSequence] = None
        self._all_creative_insights: List[str] = []  # 历史所有创意洞察

        # ========== 远距关联网络 ==========
        # 存储 REM 阶段建立的跨记忆关联: {(mem_id_a, mem_id_b): strength}
        self._distant_associations: Dict[Tuple[str, str], float] = {}

        # ========== Schema（泛化记忆）缓存 ==========
        # NREM 阶段生成的泛化记忆，以 {schema_key: description} 形式存储
        self._schemas: Dict[str, str] = {}

        # ========== 统计计数器 ==========
        self._stats = {
            'total_sleep_sessions': 0,
            'total_sleep_cycles': 0,
            'total_memories_processed': 0,
            'total_memories_consolidated': 0,
            'total_memories_generalized': 0,
            'total_fear_extinctions': 0,
            'total_creative_insights': 0,
            'total_distant_associations': 0,
            'total_nrem_events': 0,
            'total_rem_events': 0,
        }

        # ========== 后台监控线程 ==========
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_flag: bool = False
        self._lock = threading.RLock()  # 可重入锁，防止递归死锁

    # ========================================================================
    # 记忆源注册
    # ========================================================================

    def register_memory_source(self, memory_source) -> None:
        """
        注册记忆源（通常是 CA3EpisodicMemory 实例）
        
        梦境系统通过此引用读取和修改记忆。记忆源必须提供:
        - memories: OrderedDict[str, EpisodicMemory] - 记忆存储
        - store() / recall() 等方法（可选，系统只直接操作 memories 属性）
        
        Args:
            memory_source: CA3EpisodicMemory 或兼容的记忆库实例
        """
        self._memory_source = memory_source
        logger.info("[Dream] 记忆源已注册")

    def _get_all_memories(self) -> Dict:
        """获取所有记忆的字典（安全访问记忆源）"""
        if self._memory_source is None:
            return {}
        return getattr(self._memory_source, 'memories', {})

    # ========================================================================
    # 睡眠调度管理
    # ========================================================================

    def start_monitoring(self) -> None:
        """
        启动后台睡眠监控线程
        
        当设备空闲超过 `idle_trigger_minutes` 后自动触发睡眠周期。
        监控线程以守护线程方式运行，不会阻塞主线程退出。
        """
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("[Dream] 监控线程已在运行")
            return

        self._stop_flag = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="dream-sleep-monitor",
        )
        self._monitor_thread.start()
        logger.info("[Dream] 后台睡眠监控已启动")

    def stop_monitoring(self) -> None:
        """停止后台睡眠监控线程"""
        self._stop_flag = True
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=10.0)
            self._monitor_thread = None
        logger.info("[Dream] 后台睡眠监控已停止")

    def record_activity(self) -> None:
        """
        记录用户活动（重置空闲计时器）
        
        每次收到用户输入时调用，防止在活跃期间误触发睡眠。
        如果当前正在睡眠，则不会中断当前周期（睡眠不可被打断）。
        """
        self._last_activity_time = time.time()

    def _monitor_loop(self) -> None:
        """后台空闲监控循环"""
        check_interval_s = 30  # 每30秒检查一次
        while not self._stop_flag:
            # 如果正在睡眠，跳过检查
            if self._is_sleeping:
                time.sleep(check_interval_s)
                continue

            idle_duration = time.time() - self._last_activity_time
            trigger_threshold_s = self.config.idle_trigger_minutes * 60.0

            if idle_duration >= trigger_threshold_s:
                logger.info(
                    f"[Dream] 空闲 {idle_duration / 60:.1f} 分钟，触发睡眠周期"
                )
                try:
                    self.run_full_sleep_cycle()
                except Exception as e:
                    logger.error(f"[Dream] 睡眠周期执行失败: {e}")

            time.sleep(check_interval_s)

    # ========================================================================
    # 完整睡眠周期
    # ========================================================================

    def run_full_sleep_cycle(
        self,
        depth: Optional[float] = None,
        num_cycles: Optional[int] = None,
    ) -> DreamSequence:
        """
        执行一次完整的睡眠周期
        
        典型人类睡眠结构（单周期 ~90分钟）:
          N1 (浅睡) → N2 (纺锤波) → N3 (慢波/SWS) → N2 → REM
        
        本系统简化为: NREM_LIGHT → NREM_DEEP → REM（每周期重复）
        
        Args:
            depth: 睡眠深度（0.0-1.0），控制每周期处理的记忆数量比例。
                   None 时根据空闲时长自动计算（空闲越久，深度越高）。
            num_cycles: 周期数。None 时使用配置默认值（4）。
        
        Returns:
            DreamSequence: 本次梦境的完整记录
        """
        with self._lock:
            if self._is_sleeping:
                logger.warning("[Dream] 已在睡眠中，跳过重复触发")
                return self._current_dream or DreamSequence()

            self._is_sleeping = True
            try:
                # 计算参数
                if depth is None:
                    idle_minutes = (time.time() - self._last_activity_time) / 60.0
                    # 空闲30分钟→depth=0.3，空闲2小时→depth=0.8，空闲8小时→depth=1.0
                    depth = min(1.0, 0.3 + 0.7 * min(idle_minutes / 480.0, 1.0))

                if num_cycles is None:
                    num_cycles = self.config.typical_sleep_cycles

                # 初始化梦境序列
                dream = DreamSequence(total_cycles=num_cycles)
                self._current_dream = dream

                memories = self._get_all_memories()
                if not memories:
                    logger.info("[Dream] 无记忆可处理，跳过睡眠周期")
                    return dream

                total_memories_in_session = 0

                # ========== 逐周期执行 ==========
                for cycle_idx in range(num_cycles):
                    self._current_cycle = cycle_idx + 1
                    cycle_processed = 0

                    # --- 阶段1: NREM 浅睡期 ---
                    self._current_phase = SleepPhase.NREM_LIGHT
                    nrem_count = self._run_nrem_light(
                        memories, dream, depth, total_memories_in_session,
                    )
                    cycle_processed += nrem_count
                    total_memories_in_session += nrem_count

                    # 检查是否超出总处理量限制
                    if total_memories_in_session >= self.config.max_total_memories_per_sleep:
                        logger.info(f"[Dream] 已达到总处理量上限，提前结束")
                        break

                    # --- 阶段2: NREM 深睡期 ---
                    self._current_phase = SleepPhase.NREM_DEEP
                    nrem_deep_count = self._run_nrem_deep(
                        memories, dream, depth, total_memories_in_session,
                    )
                    cycle_processed += nrem_deep_count
                    total_memories_in_session += nrem_deep_count

                    if total_memories_in_session >= self.config.max_total_memories_per_sleep:
                        logger.info(f"[Dream] 已达到总处理量上限，提前结束")
                        break

                    # --- 阶段3: REM 期 ---
                    self._current_phase = SleepPhase.REM
                    rem_count = self._run_rem(
                        memories, dream, depth, total_memories_in_session,
                    )
                    cycle_processed += rem_count
                    total_memories_in_session += rem_count

                    logger.debug(
                        f"[Dream] 周期 {cycle_idx + 1}/{num_cycles} 完成: "
                        f"处理 {cycle_processed} 条记忆"
                    )

                # ========== 完成统计 ==========
                dream.memories_processed = total_memories_in_session
                dream.phase_distribution = self._count_phase_distribution(dream)

                # 保存梦境记录
                self._dream_sequences.append(dream)
                if len(self._dream_sequences) > 100:
                    # 只保留最近100条梦境记录
                    self._dream_sequences = self._dream_sequences[-100:]

                # 更新全局统计
                self._stats['total_sleep_sessions'] += 1
                self._stats['total_sleep_cycles'] += dream.total_cycles
                self._stats['total_memories_processed'] += dream.memories_processed

                # 输出重要洞察到日志
                if dream.creative_insights:
                    logger.info(
                        f"[Dream] 本次睡眠产生 {len(dream.creative_insights)} 条创意洞察"
                    )
                    for insight in dream.creative_insights[:3]:
                        logger.info(f"[Dream] 💡 {insight}")

                # 清理
                self._current_dream = None

                logger.info(
                    f"[Dream] 睡眠完成: {dream.total_cycles} 周期, "
                    f"{dream.memories_processed} 条记忆, "
                    f"{len(dream.creative_insights)} 条洞察"
                )
                return dream
            except Exception as e:
                logger.error(f"[Dream] sleep cycle failed: {e}")
                return self._current_dream or DreamSequence()
            finally:
                self._is_sleeping = False

    def trigger_sleep(self, depth: float = 0.5, num_cycles: int = 4) -> DreamSequence:
        """
        手动触发一次睡眠周期（供外部主动调用）
        
        Args:
            depth: 睡眠深度 0.0-1.0
            num_cycles: 周期数
        
        Returns:
            DreamSequence: 梦境记录
        """
        return self.run_full_sleep_cycle(depth=depth, num_cycles=num_cycles)

    # ========================================================================
    # NREM 浅睡期 - 记忆重激活与固化
    # ========================================================================

    def _run_nrem_light(
        self,
        memories: Dict,
        dream: DreamSequence,
        depth: float,
        already_processed: int,
    ) -> int:
        """
        NREM 浅睡期: 记忆重激活 + 初步固化
        
        按重要性排序，系统性回放记忆，将符合条件的短期记忆向中期/长期迁移。
        重要性排序: is_core > emotion_intensity > recall_count > activation_strength
        
        Args:
            memories: 所有记忆字典
            dream: 当前梦境序列
            depth: 睡眠深度
            already_processed: 本轮已处理记忆数
        
        Returns:
            本阶段处理的记忆数量
        """
        # 计算本阶段处理量
        remaining_budget = self.config.max_total_memories_per_sleep - already_processed
        budget = min(
            self.config.nrem_max_memories_per_cycle,
            int(self.config.nrem_max_memories_per_cycle * depth),
            remaining_budget,
        )
        if budget <= 0:
            return 0

        # 筛选候选记忆（优先处理短期和中期记忆）
        candidates = []
        for mem_id, mem in memories.items():
            tier = getattr(mem, 'tier', MemoryTier.SHORT_TERM)
            # 浅睡期主要处理短期记忆，兼顾中期
            if tier in (MemoryTier.SHORT_TERM, MemoryTier.MID_TERM):
                candidates.append((mem_id, mem))

        if not candidates:
            return 0

        # 按重要性排序
        candidates.sort(key=lambda x: self._memory_importance_score(x[1]), reverse=True)

        processed = 0
        consolidation_manager = (
            getattr(self._memory_source, 'consolidation_manager', None)
            if self._memory_source else None
        )

        for mem_id, mem in candidates[:budget]:
            # --- 记忆重激活 ---
            # 增强激活强度（模拟系统性回放中的突触增强）
            activation_boost = 1.03 if getattr(mem, 'is_core', False) else 1.01
            current_activation = getattr(mem, 'activation_strength', 0.5)
            mem.activation_strength = min(
                current_activation * activation_boost, 2.0
            )

            # --- 记忆固化检查 ---
            promoted = False
            if consolidation_manager is not None:
                target_tier = consolidation_manager.should_promote(mem)
                if target_tier is not None:
                    old_tier = mem.tier
                    mem.tier = target_tier
                    # 固化时增强激活强度
                    mem.activation_strength = min(
                        mem.activation_strength * 1.2, 2.0
                    )
                    promoted = True
                    self._stats['total_memories_consolidated'] += 1

                    # 记录事件
                    event = DreamEvent(
                        memory_ids=[mem_id],
                        phase=SleepPhase.NREM_LIGHT,
                        event_type=DreamEventType.CONSOLIDATE,
                        result=(
                            f"记忆 {mem_id[:8]} 从 "
                            f"{old_tier.name} 固化至 {target_tier.name}"
                        ),
                        importance=0.7,
                    )
                    dream.events.append(event)
                    self._stats['total_nrem_events'] += 1

            if not promoted:
                # 仅重激活事件
                event = DreamEvent(
                    memory_ids=[mem_id],
                    phase=SleepPhase.NREM_LIGHT,
                    event_type=DreamEventType.REACTIVATE,
                    result=f"记忆 {mem_id[:8]} 已重激活",
                    importance=0.2,
                )
                dream.events.append(event)
                self._stats['total_nrem_events'] += 1

            processed += 1

        return processed

    # ========================================================================
    # NREM 深睡期 - 记忆泛化与稳定化
    # ========================================================================

    def _run_nrem_deep(
        self,
        memories: Dict,
        dream: DreamSequence,
        depth: float,
        already_processed: int,
    ) -> int:
        """
        NREM 深睡期（慢波睡眠 SWS）: 记忆泛化 + 长期记忆稳定化
        
        深睡期是记忆泛化的黄金窗口:
        - 从相似记忆中提取共同模式，生成"schema"（泛化记忆）
        - 强化现有长期记忆的稳定性
        - 模式补全: 填充记忆网络中的空缺
        
        Args:
            memories: 所有记忆字典
            dream: 当前梦境序列
            depth: 睡眠深度
            already_processed: 本轮已处理记忆数
        
        Returns:
            本阶段处理的记忆数量
        """
        processed = 0

        # --- 1. 长期记忆稳定化 ---
        ltm_memories = [
            (mid, m) for mid, m in memories.items()
            if getattr(m, 'tier', MemoryTier.SHORT_TERM) == MemoryTier.LONG_TERM
        ]
        stabilize_count = min(len(ltm_memories), int(20 * depth))
        for mem_id, mem in ltm_memories[:stabilize_count]:
            boost = self.config.nrem_stabilization_boost
            current_activation = getattr(mem, 'activation_strength', 0.5)
            mem.activation_strength = min(current_activation * boost, 2.0)

            event = DreamEvent(
                memory_ids=[mem_id],
                phase=SleepPhase.NREM_DEEP,
                event_type=DreamEventType.STABILIZE,
                result=f"长期记忆 {mem_id[:8]} 稳定化增强 (×{boost})",
                importance=0.3,
            )
            dream.events.append(event)
            self._stats['total_nrem_events'] += 1
            processed += 1

        # --- 2. 记忆泛化 ---
        # 从相似的中期/长期记忆中提取共同模式
        all_mem_ids = list(memories.keys())
        generalization_budget = min(10, int(5 * depth))
        generalizations_done = 0

        for i in range(len(all_mem_ids)):
            if generalizations_done >= generalization_budget:
                break

            seed_id = all_mem_ids[i]
            seed_mem = memories[seed_id]
            seed_tier = getattr(seed_mem, 'tier', MemoryTier.SHORT_TERM)

            # 只对中期和长期记忆进行泛化
            if seed_tier in (MemoryTier.SHORT_TERM,):
                continue

            # 寻找相似记忆
            similar_group = self._find_similar_memories(
                seed_id, memories, threshold=self.config.nrem_generalization_threshold
            )

            if len(similar_group) >= self.config.nrem_generalization_threshold:
                # 提取共同模式并生成 schema
                schema_key, schema_desc = self._extract_schema(
                    [seed_id] + [s[0] for s in similar_group], memories
                )

                if schema_key and schema_key not in self._schemas:
                    self._schemas[schema_key] = schema_desc
                    self._stats['total_memories_generalized'] += 1
                    generalizations_done += 1

                    event = DreamEvent(
                        memory_ids=[seed_id] + [s[0] for s in similar_group],
                        phase=SleepPhase.NREM_DEEP,
                        event_type=DreamEventType.GENERALIZE,
                        result=f"泛化生成 schema: {schema_desc}",
                        importance=0.8,
                    )
                    dream.events.append(event)
                    self._stats['total_nrem_events'] += 1
                    processed += len(similar_group)

                    logger.info(f"[Dream] 🧠 泛化: {schema_desc}")

        return processed

    # ========================================================================
    # REM 期 - 创造性重组 + 情绪加工
    # ========================================================================

    def _run_rem(
        self,
        memories: Dict,
        dream: DreamSequence,
        depth: float,
        already_processed: int,
    ) -> int:
        """
        REM 期: 创造性重组 + 远距关联 + 情绪加工
        
        REM 是最"梦幻"的阶段，特征:
        - 记忆元素的自由组合（超现实融合）
        - 不相关记忆之间的关联形成
        - 恐惧/焦虑等负性情绪的逐步消退
        - 创意洞察的产生
        
        Args:
            memories: 所有记忆字典
            dream: 当前梦境序列
            depth: 睡眠深度
            already_processed: 本轮已处理记忆数
        
        Returns:
            本阶段处理的记忆数量
        """
        if not memories:
            return 0

        processed = 0
        all_mem_ids = list(memories.keys())

        # --- 1. 恐惧消退 ---
        fear_count = self._process_fear_extinction(memories, dream)
        processed += fear_count

        # --- 2. 情绪加工（正面增强 + 负面衰减）---
        emotion_count = self._process_emotion_consolidation(memories, dream)
        processed += emotion_count

        # --- 3. 创造性重组 ---
        recomb_count = self._process_creative_recombination(
            memories, dream, depth
        )
        processed += recomb_count

        # --- 4. 远距关联形成 ---
        assoc_count = self._process_distant_associations(
            memories, dream, depth
        )
        processed += assoc_count

        # --- 5. 梦境序列生成（可选）---
        self._generate_dream_sequence_inner(memories, dream, depth)

        self._stats['total_rem_events'] += (
            fear_count + emotion_count + recomb_count + assoc_count
        )

        return processed

    # ========================================================================
    # 梦境序列生成（REM 中的自由联想）
    # ========================================================================

    def generate_dream_sequence(
        self,
        memories: Optional[Dict] = None,
        duration_cycles: int = 5,
    ) -> DreamSequence:
        """
        生成一段梦境序列（外部调用接口）
        
        模拟 REM 睡眠中的自由联想过程:
        1. 从一条近期记忆作为种子出发
        2. 沿关联链行走，逐渐增加随机性
        3. 偶尔跳跃到高情绪强度记忆（梦境中的"突转"）
        4. 混合不同记忆的元素（超现实融合）
        
        Args:
            memories: 记忆字典，为 None 时从记忆源获取
            duration_cycles: 梦境持续周期数（影响序列长度）
        
        Returns:
            DreamSequence: 模拟的梦境序列
        """
        if memories is None:
            memories = self._get_all_memories()

        dream = DreamSequence(total_cycles=duration_cycles)
        self._generate_dream_sequence_inner(memories, dream, 1.0)
        dream.memories_processed = len(set(
            mid for event in dream.events for mid in event.memory_ids
        ))
        dream.new_associations = sum(
            1 for event in dream.events
            if event.event_type in (DreamEventType.COMBINE, DreamEventType.ASSOCIATE)
        )
        dream.creative_insights = [
            event.result for event in dream.events
            if event.event_type == DreamEventType.INSIGHT
        ]
        dream.phase_distribution = self._count_phase_distribution(dream)

        return dream

    def _generate_dream_sequence_inner(
        self,
        memories: Dict,
        dream: DreamSequence,
        depth: float,
    ) -> None:
        """
        梦境序列生成的内部实现
        
        使用随机游走算法模拟梦境中的自由联想:
        - 起始: 选择一条近期、高激活的记忆作为种子
        - 行走: 沿 causal_links 关联链前进，每步增加随机性
        - 跳跃: 以一定概率跳转到高情绪强度记忆
        - 融合: 混合相邻记忆的语义元素
        """
        if not memories:
            return

        all_mem_ids = list(memories.keys())
        if not all_mem_ids:
            return

        # 选择种子记忆: 优先选择近期、高激活、有情绪的记忆
        seed_candidates = sorted(
            all_mem_ids,
            key=lambda mid: self._dream_seed_score(memories[mid]),
            reverse=True,
        )
        seed_id = seed_candidates[0] if seed_candidates else all_mem_ids[0]

        visited = set()
        current_id = seed_id
        sequence_length = int(10 * depth)

        for step in range(sequence_length):
            if current_id not in memories:
                break

            current_mem = memories[current_id]

            # --- 随机行走: 选择下一步 ---
            next_id = self._dream_step(
                current_id, current_mem, memories, all_mem_ids, depth
            )

            if next_id and next_id != current_id and next_id in memories:
                # 记录"场景转换"事件
                blend_desc = self._blend_memory_elements(
                    current_mem, memories[next_id]
                )

                if random.random() < self.config.rem_surreal_blend_probability:
                    # 超现实融合: 创造性跳跃
                    event = DreamEvent(
                        memory_ids=[current_id, next_id],
                        phase=SleepPhase.REM,
                        event_type=DreamEventType.COMBINE,
                        result=f"梦境融合: {blend_desc}",
                        importance=0.6,
                    )
                else:
                    # 普通联想转移
                    pointer = getattr(memories[next_id], 'semantic_pointer', None) or getattr(memories[next_id], 'content', '') or '未知'
                    event = DreamEvent(
                        memory_ids=[current_id, next_id],
                        phase=SleepPhase.REM,
                        event_type=DreamEventType.REACTIVATE,
                        result=f"梦境转移: {str(pointer)[:30]}",
                        importance=0.2,
                    )

                dream.events.append(event)
                visited.add(current_id)
                visited.add(next_id)
                current_id = next_id
            else:
                # 无法继续行走，随机跳转
                unvisited = [mid for mid in all_mem_ids if mid not in visited]
                if unvisited:
                    current_id = random.choice(unvisited)
                else:
                    break

            # 事件数限制
            if len(dream.events) >= self.config.dream_sequence_max_events:
                break

    def _dream_seed_score(self, mem) -> float:
        """
        计算记忆作为梦境种子的适合度
        
        梦境更倾向于选择:
        - 近期记忆（timestamp越大越好）
        - 高情绪强度
        - 高激活强度
        """
        emotion = getattr(mem, 'emotion_intensity', 0.0)
        activation = getattr(mem, 'activation_strength', 0.5)
        # 时间新鲜度计算（安全处理不同量级的时间戳）
        timestamp_ms = getattr(mem, 'timestamp', 0)
        if timestamp_ms > 1e12:  # 毫秒级时间戳
            age_hours = (time.time() * 1000 - timestamp_ms) / 3600000.0
        elif timestamp_ms > 1e9:  # 秒级时间戳
            age_hours = (time.time() - timestamp_ms) / 3600.0
        else:
            age_hours = 999.0  # 未知时间戳，视为非常陈旧
        recency = max(0.0, 1.0 - age_hours / 720.0)  # 30天内线性衰减
        return 0.4 * recency + 0.3 * emotion + 0.3 * activation

    def _dream_step(
        self,
        current_id: str,
        current_mem,
        memories: Dict,
        all_ids: List[str],
        depth: float,
    ) -> Optional[str]:
        """
        梦境中的"一步" - 选择下一个记忆
        
        策略:
        1. 优先沿 causal_links 前进（关联链行走）
        2. 以 creative_jump_probability 跳转到高情绪记忆
        3. 否则随机选择一个记忆（随机游走）
        """
        # 策略1: 沿关联链
        causal_links = getattr(current_mem, 'causal_links', [])
        linked_ids = []
        for link in causal_links:
            if 'linked_to_' in str(link):
                # 从 "linked_to_{id}_{strength}" 提取目标ID
                parts = str(link).replace('linked_to_', '').rsplit('_', 1)
                if len(parts) >= 1:
                    target_id = parts[0]
                    if target_id in memories:
                        linked_ids.append(target_id)

        if linked_ids and random.random() > self.config.rem_creative_jump_probability:
            return random.choice(linked_ids)

        # 策略2: 跳跃到高情绪记忆
        if random.random() < self.config.rem_creative_jump_probability:
            # 按情绪强度排序，选择一个高情绪记忆
            emotional_mems = sorted(
                all_ids,
                key=lambda mid: getattr(memories[mid], 'emotion_intensity', 0.0),
                reverse=True,
            )
            top_emotional = emotional_mems[:max(3, len(emotional_mems) // 10)]
            if top_emotional:
                return random.choice(top_emotional)

        # 策略3: 随机游走
        return random.choice(all_ids) if all_ids else None

    def _blend_memory_elements(self, mem_a, mem_b) -> str:
        """
        融合两个记忆的元素，产生"超现实"描述
        
        模拟梦境中将不同记忆的元素混合在一起的现象。
        例如: "Python编程" + "艺术创作" → "用Python画画的创意编程"
        """
        summary_a = getattr(mem_a, 'semantic_summary', '') or getattr(mem_a, 'content', '')[:50]
        summary_b = getattr(mem_b, 'semantic_summary', '') or getattr(mem_b, 'content', '')[:50]
        pointer_a = getattr(mem_a, 'semantic_pointer', '')[:20]
        pointer_b = getattr(mem_b, 'semantic_pointer', '')[:20]

        # 如果有语义摘要，使用摘要融合
        if summary_a and summary_b:
            return f"{summary_a[:20]} × {summary_b[:20]} → 融合记忆"
        elif pointer_a and pointer_b:
            return f"{pointer_a} × {pointer_b} → 梦境混合"
        else:
            return "抽象元素融合"

    # ========================================================================
    # 恐惧消退 (REM)
    # ========================================================================

    def _process_fear_extinction(
        self,
        memories: Dict,
        dream: DreamSequence,
    ) -> int:
        """
        REM 恐惧消退: 系统性降低恐惧类记忆的情绪强度
        
        神经科学依据:
        - REM 睡眠是恐惧消退学习的关键窗口
        - 杏仁核在REM期间降低反应性
        - 前额叶皮层在REM期间增强调控
        - 效果: 保留语义内容，降低情绪负荷
        
        实现:
        - emotion_intensity *= 0.95（每REM周期）
        - 生存相关恐惧保留最低强度 0.1（不会完全消退）
        """
        count = 0
        extinction_rate = self.config.rem_fear_extinction_rate
        minimum = self.config.rem_fear_minimum_intensity

        for mem_id, mem in memories.items():
            emotion_type = getattr(mem, 'emotion_type', '').lower()
            emotion_intensity = getattr(mem, 'emotion_intensity', 0.0)

            if emotion_intensity <= 0.0:
                continue

            # 判断是否为恐惧/焦虑类记忆
            is_fear = any(
                keyword in emotion_type
                for keyword in self.config.survival_related_emotions
            )
            is_negative = emotion_type in (
                'negative', 'fear', 'anger', 'sadness', 'anxiety',
                '负面', '恐惧', '愤怒', '悲伤', '焦虑', 'worried',
            )

            if not is_fear and not is_negative:
                continue

            # 计算消退后的情绪强度
            old_intensity = emotion_intensity
            new_intensity = emotion_intensity * extinction_rate

            # 判断是否为生存相关恐惧（保留最低强度）
            is_survival = is_fear  # 恐惧和危险类情绪视为生存相关
            if is_survival:
                new_intensity = max(new_intensity, minimum)

            mem.emotion_intensity = round(new_intensity, 4)
            count += 1
            self._stats['total_fear_extinctions'] += 1

            # 记录消退事件（仅当消退幅度较大时记录）
            if old_intensity - new_intensity > 0.01:
                event = DreamEvent(
                    memory_ids=[mem_id],
                    phase=SleepPhase.REM,
                    event_type=DreamEventType.EXTINCT,
                    result=(
                        f"恐惧消退: {mem_id[:8]} 情绪强度 "
                        f"{old_intensity:.3f} → {new_intensity:.3f}"
                    ),
                    importance=0.4 + (0.3 if is_survival else 0.0),
                )
                dream.events.append(event)

        return count

    # ========================================================================
    # 情绪巩固 (REM)
    # ========================================================================

    def _process_emotion_consolidation(
        self,
        memories: Dict,
        dream: DreamSequence,
    ) -> int:
        """
        REM 情绪巩固: 正面情绪增强，负面情绪衰减
        
        神经科学依据:
        - REM 睡眠优先巩固情绪记忆
        - 正面情绪记忆得到选择性增强
        - 负面情绪（非恐惧）的过度反应被调节
        """
        count = 0

        for mem_id, mem in memories.items():
            emotion_type = getattr(mem, 'emotion_type', '').lower()
            emotion_intensity = getattr(mem, 'emotion_intensity', 0.0)

            if emotion_intensity <= 0.01:
                continue

            is_positive = emotion_type in (
                'positive', 'joy', 'happiness', 'love', 'excitement',
                '正面', '快乐', '幸福', '爱', '兴奋', 'neutral',
            )
            is_negative = emotion_type in (
                'negative', 'fear', 'anger', 'sadness', 'anxiety',
                '负面', '恐惧', '愤怒', '悲伤', '焦虑', 'worried',
            )

            if is_positive:
                # 正面情绪增强
                old = emotion_intensity
                mem.emotion_intensity = min(
                    emotion_intensity * self.config.positive_emotion_consolidation_rate,
                    1.0,
                )
                if abs(mem.emotion_intensity - old) > 0.001:
                    count += 1
            elif is_negative:
                # 负面情绪衰减（非恐惧部分已在恐惧消退中处理）
                if emotion_type not in self.config.survival_related_emotions:
                    mem.emotion_intensity = max(
                        emotion_intensity * self.config.negative_emotion_decay_rate,
                        0.01,
                    )
                    count += 1

        return count

    # ========================================================================
    # 创造性重组 (REM)
    # ========================================================================

    def _process_creative_recombination(
        self,
        memories: Dict,
        dream: DreamSequence,
        depth: float,
    ) -> int:
        """
        REM 创造性重组: 将不相关的记忆元素融合，产生新洞察
        
        神经科学依据:
        - REM 睡眠中海马体-新皮层的信息流动逆转
        - 胼胝体连接增强，允许远距脑区协同
        - 去甲肾上腺素水平极低，降低注意力约束，允许自由联想
        
        实现:
        1. 随机选择 2-3 条不相关的记忆
        2. 提取各自的语义元素
        3. 生成融合描述（创意洞察）
        4. 存储为特殊的中期记忆
        """
        all_ids = list(memories.keys())
        if len(all_ids) < 2:
            return 0

        budget = min(
            self.config.rem_max_recombinations,
            max(1, int(self.config.rem_max_recombinations * depth)),
        )
        count = 0

        for _ in range(budget):
            # 随机选择 2-3 条记忆
            num_combine = random.choice([2, 3])
            if len(all_ids) >= num_combine:
                selected_ids = random.sample(all_ids, num_combine)
            else:
                selected_ids = all_ids

            selected_mems = [memories[mid] for mid in selected_ids]

            # 生成创意洞察
            insight = self._generate_creative_insight(selected_mems)
            if insight:
                count += 1
                self._stats['total_creative_insights'] += 1
                dream.creative_insights.append(insight)
                self._all_creative_insights.append(insight)

                # 限制历史洞察数量
                if len(self._all_creative_insights) > 1000:
                    self._all_creative_insights = self._all_creative_insights[-500:]

                event = DreamEvent(
                    memory_ids=selected_ids,
                    phase=SleepPhase.REM,
                    event_type=DreamEventType.INSIGHT,
                    result=insight,
                    importance=0.9,
                )
                dream.events.append(event)

                logger.info(f"[Dream] 💡 创意洞察: {insight}")

                # 存储为特殊的关联记忆（强化两两之间的连接）
                for i in range(len(selected_ids)):
                    for j in range(i + 1, len(selected_ids)):
                        pair = (selected_ids[i], selected_ids[j])
                        self._distant_associations[pair] = min(
                            self._distant_associations.get(pair, 0.0) + 0.3,
                            1.0,
                        )

        return count

    def _generate_creative_insight(self, memories: List) -> Optional[str]:
        """
        从多条记忆中生成创意洞察
        
        提取各记忆的语义元素，尝试识别有意义的组合。
        例如:
        - 记忆A: "Python" + 记忆B: "艺术" → "Python + 艺术 = 创意编程"
        - 记忆A: "用户喜欢咖啡" + 记忆B: "用户是程序员" → "程序员偏好咖啡"
        
        Args:
            memories: 参与重组的记忆列表（2-3条）
        
        Returns:
            创意洞察描述，或 None（无法生成有意义洞察时）
        """
        if not memories:
            return None

        elements = []
        for mem in memories:
            # 收集记忆的语义元素
            summary = getattr(mem, 'semantic_summary', '').strip()
            content = getattr(mem, 'content', '').strip()
            entities = getattr(mem, 'key_entities', '').strip()
            pointer = getattr(mem, 'semantic_pointer', '').strip()

            # 优先使用实体 > 摘要 > 内容
            text = entities or summary or content or pointer
            if text:
                # 截取前50字符作为元素
                elements.append(text[:50])

        if len(elements) < 2:
            return None

        # 生成洞察描述
        if len(elements) == 2:
            insight = f"{elements[0][:20]} × {elements[1][:20]} → 潜在创意关联"
        else:
            insight = (
                f"{elements[0][:15]} × {elements[1][:15]} × {elements[2][:15]} "
                f"→ 多源融合洞察"
            )

        # 过滤过于笼统的洞察
        generic_terms = {'token', 'context', 'null', 'empty', 'none', ''}
        unique_elements = [e for e in elements if e.lower() not in generic_terms]
        if len(unique_elements) < 2:
            return None

        return insight

    # ========================================================================
    # 远距关联形成 (REM)
    # ========================================================================

    def _process_distant_associations(
        self,
        memories: Dict,
        dream: DreamSequence,
        depth: float,
    ) -> int:
        """
        REM 远距关联: 在看似无关的记忆间建立联系
        
        神经科学依据:
        - REM 睡眠中默认模式网络（DMN）高度活跃
        - 海马体 CA3 区的自联想网络允许自由模式补全
        - 去甲肾上腺素降低使得注意力约束放宽
        
        实现:
        - 选择语义距离较远的记忆对
        - 建立弱关联（初始强度较低，需多次REM周期才能巩固）
        - 关联存储在 _distant_associations 字典中
        """
        all_ids = list(memories.keys())
        if len(all_ids) < 2:
            return 0

        budget = min(
            self.config.rem_distant_association_max,
            max(1, int(self.config.rem_distant_association_max * depth)),
        )
        count = 0
        attempts = 0
        max_attempts = budget * 10  # 防止无限循环

        while count < budget and attempts < max_attempts:
            attempts += 1

            # 随机选择两个不同的记忆
            id_a, id_b = random.sample(all_ids, 2)
            pair = (id_a, id_b)

            # 跳过已有关联
            if pair in self._distant_associations:
                continue

            # 检查是否"远距"（通过共享实体判断）
            mem_a = memories[id_a]
            mem_b = memories[id_b]
            entities_a = set(
                getattr(mem_a, 'key_entities', '').split('|')
            ) - {''}
            entities_b = set(
                getattr(mem_b, 'key_entities', '').split('|')
            ) - {''}

            # 如果完全没有共享实体，则认为是"远距"的
            if entities_a and entities_b and not entities_a & entities_b:
                # 建立弱关联
                initial_strength = 0.1 + 0.2 * random.random()
                self._distant_associations[pair] = round(initial_strength, 4)
                count += 1
                self._stats['total_distant_associations'] += 1

                desc_a = getattr(mem_a, 'semantic_pointer', id_a[:8])[:20]
                desc_b = getattr(mem_b, 'semantic_pointer', id_b[:8])[:20]

                event = DreamEvent(
                    memory_ids=[id_a, id_b],
                    phase=SleepPhase.REM,
                    event_type=DreamEventType.ASSOCIATE,
                    result=f"远距关联: [{desc_a}] ↔ [{desc_b}] (强度: {initial_strength:.2f})",
                    importance=0.6,
                )
                dream.events.append(event)
            elif not entities_a or not entities_b:
                # 没有实体信息的记忆也可以建立关联
                initial_strength = 0.05 + 0.1 * random.random()
                self._distant_associations[pair] = round(initial_strength, 4)
                count += 1
                self._stats['total_distant_associations'] += 1

        return count

    # ========================================================================
    # 记忆泛化辅助方法 (NREM)
    # ========================================================================

    def _find_similar_memories(
        self,
        seed_id: str,
        memories: Dict,
        threshold: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        寻找与种子记忆相似的其他记忆
        
        相似度判断基于:
        - 共享关键实体
        - 情感标签一致性
        - 语义指针的文本相似度（简单字符重叠）
        
        Args:
            seed_id: 种子记忆 ID
            memories: 所有记忆字典
            threshold: 最少需要的相似记忆数（用于判断是否触发泛化）
        
        Returns:
            相似记忆列表: [(mem_id, similarity_score), ...]
        """
        if seed_id not in memories:
            return []

        seed = memories[seed_id]
        seed_entities = set(
            getattr(seed, 'key_entities', '').split('|')
        ) - {''}
        seed_emotion = getattr(seed, 'emotion_type', '')
        seed_pointer = getattr(seed, 'semantic_pointer', '').lower()

        similar = []
        for mid, mem in memories.items():
            if mid == seed_id:
                continue

            score = 0.0

            # 1. 实体重叠（最重要的相似度指标）
            mem_entities = set(
                getattr(mem, 'key_entities', '').split('|')
            ) - {''}
            if seed_entities and mem_entities:
                overlap = seed_entities & mem_entities
                if overlap:
                    # Jaccard 相似度
                    union = seed_entities | mem_entities
                    score += len(overlap) / len(union)

            # 2. 情感标签一致性
            mem_emotion = getattr(mem, 'emotion_type', '')
            if seed_emotion and mem_emotion and seed_emotion == mem_emotion:
                score += 0.2

            # 3. 语义指针字符重叠
            mem_pointer = getattr(mem, 'semantic_pointer', '').lower()
            if seed_pointer and mem_pointer:
                common_chars = set(seed_pointer) & set(mem_pointer)
                if common_chars:
                    total_chars = set(seed_pointer) | set(mem_pointer)
                    score += 0.1 * (len(common_chars) / max(len(total_chars), 1))

            if score > 0.15:
                similar.append((mid, round(score, 4)))

        # 按相似度排序
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:10]

    def _extract_schema(
        self,
        memory_ids: List[str],
        memories: Dict,
    ) -> Tuple[Optional[str], str]:
        """
        从一组相似记忆中提取共同模式（schema）
        
        示例:
        - 输入: ["我去餐厅A吃饭", "我在餐厅B点了牛排", "昨天在餐厅C聚餐"]
        - 输出: ("外出就餐偏好", "用户经常外出就餐，涉及多家餐厅")
        
        Args:
            memory_ids: 相似记忆的 ID 列表
            memories: 所有记忆字典
        
        Returns:
            (schema_key, schema_description): 泛化记忆的键和描述
        """
        if not memory_ids:
            return None, ""

        # 收集所有记忆的语义元素
        all_summaries = []
        all_entities = []
        all_contents = []
        emotion_types = set()

        for mid in memory_ids:
            if mid not in memories:
                continue
            mem = memories[mid]

            summary = getattr(mem, 'semantic_summary', '').strip()
            content = getattr(mem, 'content', '').strip()
            entities = getattr(mem, 'key_entities', '').strip()
            emotion = getattr(mem, 'emotion_type', '').strip()

            if summary:
                all_summaries.append(summary[:50])
            if content:
                all_contents.append(content[:50])
            if entities:
                all_entities.extend(entities.split('|'))
            if emotion:
                emotion_types.add(emotion)

        # 至少需要摘要、内容或实体才能生成 schema
        if not all_summaries and not all_contents and not all_entities:
            return None, ""

        # 提取共同实体作为 schema key
        entity_counter: Dict[str, int] = {}
        for ent in all_entities:
            ent = ent.strip()
            if ent and len(ent) >= 2:
                entity_counter[ent] = entity_counter.get(ent, 0) + 1

        # 出现次数最多的实体作为 schema key
        common_entities = sorted(
            entity_counter.items(), key=lambda x: x[1], reverse=True
        )
        schema_key = (
            "schema_" + "_".join(e[0] for e in common_entities[:3])
            if common_entities
            else f"schema_{memory_ids[0][:8]}"
        )

        # 生成描述
        desc_parts = []
        if all_summaries:
            # 取前3条摘要的片段组合
            combined = " | ".join(s[:20] for s in all_summaries[:3])
            desc_parts.append(f"泛化模式: {combined}")
        if common_entities:
            frequent = [e[0] for e in common_entities[:3]]
            desc_parts.append(f"共同主题: {', '.join(frequent)}")
        if emotion_types:
            desc_parts.append(f"情绪特征: {', '.join(emotion_types)}")

        schema_desc = "; ".join(desc_parts) if desc_parts else "抽象模式"
        return schema_key, schema_desc

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _memory_importance_score(self, mem) -> float:
        """
        计算记忆的重要性分数
        
        NREM 阶段按此分数排序决定记忆处理优先级:
        is_core > emotion_intensity > recall_count > activation_strength
        """
        score = 0.0

        # 核心记忆优先（权重最高）
        if getattr(mem, 'is_core', False):
            score += 3.0

        # 情绪强度
        emotion = getattr(mem, 'emotion_intensity', 0.0)
        score += emotion * 2.0

        # 被召回次数
        recalls = getattr(mem, 'recall_count', 0)
        score += min(recalls * 0.1, 1.0)

        # 激活强度
        activation = getattr(mem, 'activation_strength', 0.5)
        score += activation * 0.5

        return score

    @staticmethod
    def _count_phase_distribution(dream: DreamSequence) -> Dict[str, int]:
        """统计梦境中各阶段的事件数"""
        dist: Dict[str, int] = {}
        for event in dream.events:
            phase_name = event.phase.value
            dist[phase_name] = dist.get(phase_name, 0) + 1
        return dist

    # ========================================================================
    # 统计与状态管理
    # ========================================================================

    def get_stats(self) -> dict:
        """
        获取梦境巩固系统的完整统计信息
        
        Returns:
            dict: 包含所有统计指标的字典
        """
        return {
            # --- 系统状态 ---
            'is_sleeping': self._is_sleeping,
            'current_phase': self._current_phase.value if self._current_phase else 'awake',
            'current_cycle': self._current_cycle,
            'idle_duration_minutes': (time.time() - self._last_activity_time) / 60.0,

            # --- 累计统计 ---
            'total_sleep_sessions': self._stats['total_sleep_sessions'],
            'total_sleep_cycles': self._stats['total_sleep_cycles'],
            'total_memories_processed': self._stats['total_memories_processed'],
            'total_memories_consolidated': self._stats['total_memories_consolidated'],
            'total_memories_generalized': self._stats['total_memories_generalized'],
            'total_fear_extinctions': self._stats['total_fear_extinctions'],
            'total_creative_insights': self._stats['total_creative_insights'],
            'total_distant_associations': self._stats['total_distant_associations'],
            'total_nrem_events': self._stats['total_nrem_events'],
            'total_rem_events': self._stats['total_rem_events'],

            # --- 网络状态 ---
            'num_distant_associations': len(self._distant_associations),
            'num_schemas': len(self._schemas),
            'num_dream_sequences': len(self._dream_sequences),
            'num_creative_insights_history': len(self._all_creative_insights),

            # --- 最近洞察 ---
            'recent_insights': self._all_creative_insights[-5:],
            'recent_schemas': list(self._schemas.items())[-5:],
        }

    def get_dream_history(self, limit: int = 10) -> List[dict]:
        """
        获取最近的梦境历史记录
        
        Args:
            limit: 返回的最大梦境数量
        
        Returns:
            梦境序列字典列表（按时间倒序）
        """
        recent = self._dream_sequences[-limit:]
        return [dream.to_dict() for dream in reversed(recent)]

    def get_association_network(self) -> Dict[str, Any]:
        """
        获取远距关联网络
        
        Returns:
            关联网络字典: {关联键: 强度}
        """
        return {
            f"{a[:8]}↔{b[:8]}": strength
            for (a, b), strength in self._distant_associations.items()
        }

    def get_schemas(self) -> Dict[str, str]:
        """
        获取所有泛化记忆（schemas）
        
        Returns:
            {schema_key: schema_description}
        """
        return dict(self._schemas)

    # ========================================================================
    # 序列化 / 反序列化
    # ========================================================================

    def get_state(self) -> dict:
        """
        获取梦境系统的完整状态（用于持久化）
        
        Returns:
            dict: 可序列化的状态字典
        """
        return {
            # 配置
            'config': {
                'idle_trigger_minutes': self.config.idle_trigger_minutes,
                'typical_sleep_cycles': self.config.typical_sleep_cycles,
                'nrem_max_memories_per_cycle': self.config.nrem_max_memories_per_cycle,
                'rem_max_recombinations': self.config.rem_max_recombinations,
                'rem_fear_extinction_rate': self.config.rem_fear_extinction_rate,
                'rem_fear_minimum_intensity': self.config.rem_fear_minimum_intensity,
                'rem_creative_jump_probability': self.config.rem_creative_jump_probability,
                'rem_surreal_blend_probability': self.config.rem_surreal_blend_probability,
                'max_total_memories_per_sleep': self.config.max_total_memories_per_sleep,
            },
            # 睡眠状态
            'is_sleeping': self._is_sleeping,
            'last_activity_time': self._last_activity_time,
            # 统计
            'stats': dict(self._stats),
            # 远距关联
            'distant_associations': {
                f"{a}|{b}": strength
                for (a, b), strength in self._distant_associations.items()
            },
            # Schemas
            'schemas': dict(self._schemas),
            # 梦境历史
            'dream_sequences': [d.to_dict() for d in self._dream_sequences[-20:]],
            # 创意洞察历史
            'creative_insights': self._all_creative_insights[-200:],
        }

    def set_state(self, state: dict) -> None:
        """
        从状态字典恢复梦境系统
        
        Args:
            state: 之前通过 get_state() 导出的状态字典
        """
        # 恢复配置（只覆盖提供的字段）
        config_data = state.get('config', {})
        if config_data:
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # 恢复睡眠状态
        self._is_sleeping = state.get('is_sleeping', False)
        self._last_activity_time = state.get('last_activity_time', time.time())

        # 恢复统计
        saved_stats = state.get('stats', {})
        for key in self._stats:
            if key in saved_stats:
                self._stats[key] = saved_stats[key]

        # 恢复远距关联
        saved_assocs = state.get('distant_associations', {})
        self._distant_associations = {}
        for key, strength in saved_assocs.items():
            parts = key.split('|')
            if len(parts) == 2:
                self._distant_associations[(parts[0], parts[1])] = strength

        # 恢复 Schemas
        self._schemas = state.get('schemas', {})

        # 恢复梦境历史
        saved_dreams = state.get('dream_sequences', [])
        self._dream_sequences = [
            DreamSequence.from_dict(d) for d in saved_dreams
        ]

        # 恢复创意洞察
        self._all_creative_insights = state.get('creative_insights', [])

        logger.info(
            f"[Dream] 状态已恢复: "
            f"{self._stats['total_sleep_sessions']} 次睡眠, "
            f"{len(self._distant_associations)} 条关联, "
            f"{len(self._schemas)} 条schema"
        )
